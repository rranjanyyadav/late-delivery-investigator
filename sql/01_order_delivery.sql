-- order_delivery: ONE ROW PER ORDER (grain = order_id).
--
-- Delivery timing and revenue for every order. Order-level metrics such as
-- on_time_delivery_rate and avg_delay_days are computed from this view, so
-- an order with 5 items still counts as one order.

CREATE OR REPLACE VIEW order_delivery AS
WITH order_totals AS (
    -- order_items has one row per item; roll it up to one row per order.
    SELECT
        order_id,
        count(*)                   AS item_count,
        count(DISTINCT seller_id)  AS seller_count,
        sum(price)                 AS items_value,
        sum(freight_value)         AS freight_value,
        sum(price + freight_value) AS order_revenue
    FROM order_items
    GROUP BY order_id
),
timing AS (
    SELECT
        o.order_id,
        o.customer_id,
        o.order_status,
        o.order_purchase_timestamp       AS purchase_ts,
        o.order_delivered_carrier_date   AS carrier_date,
        o.order_estimated_delivery_date  AS estimated_delivery_date,
        o.order_delivered_customer_date  AS delivered_date,
        -- We can only judge on-time vs late once the order has arrived.
        (o.order_status = 'delivered'
         AND o.order_delivered_customer_date IS NOT NULL) AS is_delivered
    FROM orders o
)
SELECT
    t.order_id,
    t.customer_id,
    t.order_status,
    c.customer_state,
    c.customer_city,

    -- Time dimensions (all based on the purchase date).
    t.purchase_ts,
    CAST(t.purchase_ts AS DATE)                           AS purchase_date,
    CAST(date_trunc('month', t.purchase_ts) AS DATE)      AS purchase_month,
    strftime(t.purchase_ts, '%Y') || '-Q' || quarter(t.purchase_ts) AS purchase_quarter,

    t.carrier_date,
    t.estimated_delivery_date,
    t.delivered_date,
    t.is_delivered,

    -- Days past the promised date: positive = late, zero or negative = on time.
    -- Compared at DAY level because the estimate has no time of day (00:00:00);
    -- otherwise a parcel arriving at 09:00 on the promised day would count as late.
    -- NULL for orders that have not been delivered.
    CASE WHEN t.is_delivered THEN
        date_diff('day', CAST(t.estimated_delivery_date AS DATE),
                         CAST(t.delivered_date AS DATE))
    END                                                   AS delivery_delay_days,

    CASE WHEN t.is_delivered THEN
        CAST(t.delivered_date AS DATE) > CAST(t.estimated_delivery_date AS DATE)
    END                                                   AS is_late,

    CASE WHEN t.is_delivered THEN
        date_diff('day', CAST(t.purchase_ts AS DATE), CAST(t.delivered_date AS DATE))
    END                                                   AS days_to_deliver,

    -- How many days the customer was promised at purchase. If this shrinks,
    -- the late rate can rise even when deliveries get faster.
    date_diff('day', CAST(t.purchase_ts AS DATE),
                     CAST(t.estimated_delivery_date AS DATE)) AS promised_days,

    -- Revenue = what the customer paid for the items, including freight.
    -- NULL for the ~775 orders that have no items (mostly canceled/unavailable).
    ot.item_count,
    ot.seller_count,
    ot.items_value,
    ot.freight_value,
    ot.order_revenue
FROM timing t
LEFT JOIN customers c   USING (customer_id)
LEFT JOIN order_totals ot USING (order_id);
