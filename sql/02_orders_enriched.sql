-- orders_enriched: ONE ROW PER ORDER LINE (grain = order_id + order_item_id).
--
-- Every item with its seller, product category (in English), customer and the
-- delivery outcome of the order it belongs to. Seller- and category-level
-- metrics come from this view, because one order can contain items from
-- several sellers and categories.
--
-- WARNING: order-level columns (is_late, delivery_delay_days, ...) are REPEATED
-- on every line of the same order. Count orders with count(DISTINCT order_id),
-- never count(*). Order totals (order_revenue) are deliberately NOT included
-- here, so they cannot be double-counted; use line_revenue instead.

CREATE OR REPLACE VIEW orders_enriched AS
SELECT
    -- line
    i.order_id,
    i.order_item_id,
    i.product_id,
    i.seller_id,
    i.price,
    i.freight_value,
    i.price + i.freight_value                     AS line_revenue,

    -- product: English name if we have one, else the Portuguese name, else 'unknown'
    COALESCE(tr.product_category_name_english,
             p.product_category_name,
             'unknown')                           AS category,

    -- seller
    s.seller_state,
    s.seller_city,
    i.shipping_limit_date,
    -- Did the seller hand the parcel to the carrier after its deadline?
    -- Separates "seller was slow" from "carrier was slow" when an order is late.
    CASE WHEN d.carrier_date IS NOT NULL THEN
        d.carrier_date > i.shipping_limit_date
    END                                           AS seller_shipped_late,

    -- order-level fields (repeated on each line of the order)
    d.order_status,
    d.customer_state,
    d.customer_city,
    d.purchase_ts,
    d.purchase_date,
    d.purchase_month,
    d.purchase_quarter,
    d.carrier_date,
    d.estimated_delivery_date,
    d.delivered_date,
    d.is_delivered,
    d.delivery_delay_days,
    d.is_late,
    d.days_to_deliver
FROM order_items i
JOIN order_delivery d          USING (order_id)
LEFT JOIN sellers s            USING (seller_id)
LEFT JOIN products p           USING (product_id)
LEFT JOIN category_translation tr USING (product_category_name);
