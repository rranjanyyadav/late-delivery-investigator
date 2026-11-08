"""Shared test fixtures: a TINY hand-made Olist dataset where every answer is
known in advance, built with the real loader and the real SQL views.

    orders   customer  status     purchase     estimated   delivered          note
    o1       c1 (SP)   delivered  2018-07-02   2018-07-10  2018-07-10 09:00   on time (same DAY, later hour)
    o2       c2 (RJ)   delivered  2018-07-05   2018-07-15  2018-07-20         5 days late
    o3       c3 (SP)   canceled   2018-07-06   2018-07-20  -                  not delivered
    o4       c2 (RJ)   delivered  2018-04-10   2018-04-25  2018-04-15         early (Q2)

o1 has TWO items from two sellers, so line-level and order-level answers differ:
for 2018-Q3 the correct on-time rate is 1/2 = 0.5 (orders), while averaging over
lines would give 2/3. Several tests rely on that difference.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ldi import loader  # noqa: E402

CSV_FILES = {
    "olist_customers_dataset.csv": """\
customer_id,customer_unique_id,customer_zip_code_prefix,customer_city,customer_state
c1,u1,01037,sao paulo,SP
c2,u2,20000,rio de janeiro,RJ
c3,u3,01038,sao paulo,SP
""",
    "olist_sellers_dataset.csv": """\
seller_id,seller_zip_code_prefix,seller_city,seller_state
s1,01039,sao paulo,SP
s2,80000,curitiba,PR
""",
    "olist_products_dataset.csv": """\
product_id,product_category_name,product_name_lenght,product_description_lenght,product_photos_qty,product_weight_g,product_length_cm,product_height_cm,product_width_cm
p1,beleza_saude,40,200,1,500,20,10,15
p2,,40,200,1,500,20,10,15
""",
    "product_category_name_translation.csv": """\
product_category_name,product_category_name_english
beleza_saude,health_beauty
""",
    "olist_orders_dataset.csv": """\
order_id,customer_id,order_status,order_purchase_timestamp,order_approved_at,order_delivered_carrier_date,order_delivered_customer_date,order_estimated_delivery_date
o1,c1,delivered,2018-07-02 10:00:00,2018-07-02 11:00:00,2018-07-03 12:00:00,2018-07-10 09:00:00,2018-07-10 00:00:00
o2,c2,delivered,2018-07-05 10:00:00,2018-07-05 11:00:00,2018-07-09 12:00:00,2018-07-20 15:00:00,2018-07-15 00:00:00
o3,c3,canceled,2018-07-06 10:00:00,2018-07-06 11:00:00,,,2018-07-20 00:00:00
o4,c2,delivered,2018-04-10 10:00:00,2018-04-10 11:00:00,2018-04-11 12:00:00,2018-04-15 10:00:00,2018-04-25 00:00:00
""",
    "olist_order_items_dataset.csv": """\
order_id,order_item_id,product_id,seller_id,shipping_limit_date,price,freight_value
o1,1,p1,s1,2018-07-04 00:00:00,100.00,10.00
o1,2,p2,s2,2018-07-04 00:00:00,50.00,5.00
o2,1,p1,s1,2018-07-07 00:00:00,80.00,20.00
o3,1,p2,s2,2018-07-08 00:00:00,30.00,3.00
o4,1,p1,s1,2018-04-12 00:00:00,40.00,4.00
""",
    "olist_order_payments_dataset.csv": """\
order_id,payment_sequential,payment_type,payment_installments,payment_value
o1,1,credit_card,1,165.00
o2,1,boleto,1,100.00
o3,1,credit_card,1,33.00
o4,1,credit_card,1,44.00
""",
    "olist_order_reviews_dataset.csv": """\
review_id,order_id,review_score,review_comment_title,review_comment_message,review_creation_date,review_answer_timestamp
r1,o1,5,,,2018-07-11 00:00:00,2018-07-12 00:00:00
r2,o2,1,,,2018-07-21 00:00:00,2018-07-22 00:00:00
""",
    "olist_geolocation_dataset.csv": """\
geolocation_zip_code_prefix,geolocation_lat,geolocation_lng,geolocation_city,geolocation_state
01037,-23.5,-46.6,sao paulo,SP
""",
}


@pytest.fixture(scope="session")
def tiny_db(tmp_path_factory) -> Path:
    """Build the tiny dataset with the real loader + views; return the DB path."""
    root = tmp_path_factory.mktemp("tiny")
    raw = root / "raw"
    raw.mkdir()
    for name, text in CSV_FILES.items():
        (raw / name).write_text(text, encoding="utf-8")
    db_path = root / "tiny.duckdb"
    loader.build(db_path=db_path, raw_dir=raw)
    return db_path


@pytest.fixture()
def con(tiny_db):
    from ldi.sql_guard import connect_readonly
    c = connect_readonly(tiny_db)
    yield c
    c.close()


class FakeLLM:
    """Returns scripted replies in order, and records what it was sent."""

    def __init__(self, *replies: str):
        self.replies = list(replies)
        self.calls: list[list[dict]] = []

    def chat(self, messages, json_mode=False):
        self.calls.append(messages)
        return self.replies.pop(0)
