# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Generate a large enterprise data warehouse SQLite database (500+ tables) for
complex NL2SQL benchmarking.

The warehouse simulates a real enterprise with tables across 8 domains:
  - Core Business (dim_/fact_) — star-schema sales analytics
  - Finance (fin_) — GL, invoices, AR/AP, budgets
  - HR (hr_) — employees, payroll, performance, benefits
  - Marketing (mktg_) — campaigns, leads, attribution
  - Supply Chain (sc_) — suppliers, POs, warehouses, shipments
  - Customer Support (cs_) — tickets, agents, SLAs
  - Web Analytics (wa_) — sessions, pageviews, events, funnels
  - Data Engineering (de_) — ETL jobs, data quality, lineage

Derivative tables (stg_*, raw_*, *_history, *_archive, audit_*, ref_*) are
generated programmatically to push the total past 500 tables, simulating the
discovery challenge of a real enterprise data warehouse.

Each prompt includes gold SQL for result-set matching reward.

Usage:
    python benchmark/data_gen/gen_enterprise_warehouse.py \\
        --db-path benchmark/data/enterprise_warehouse.db \\
        --output benchmark/data/nl2sql_warehouse_prompts.jsonl \\
        --seed 42
"""

import argparse
import json
import math
import os
import random
import sqlite3
from datetime import datetime, timedelta

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FIRST_NAMES = [
    "Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Henry",
    "Iris", "Jack", "Karen", "Leo", "Mia", "Noah", "Olivia", "Peter",
    "Quinn", "Rachel", "Sam", "Tara", "Uma", "Victor", "Wendy", "Xavier",
    "Yuki", "Zara", "Aiden", "Bella", "Caleb", "Daphne",
]
LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
    "Davis", "Rodriguez", "Martinez", "Wilson", "Anderson", "Taylor",
    "Thomas", "Moore", "Jackson", "Lee", "Harris", "Clark", "Lewis",
]
COUNTRIES = ["US", "UK", "DE", "FR", "CA", "AU", "JP", "BR", "IN", "KR", "MX", "IT", "ES", "NL", "SE"]
INDUSTRIES = [
    "Technology", "Finance", "Healthcare", "Retail", "Manufacturing",
    "Education", "Media", "Energy", "Telecom", "Automotive",
]
SEGMENTS = ["Enterprise", "Mid-Market", "SMB", "Startup", "Government"]
TIERS = ["Free", "Basic", "Professional", "Premium", "Enterprise"]
CHANNELS = ["Direct", "Partner", "Online", "Retail", "Wholesale", "Marketplace", "Field Sales", "Inside Sales", "OEM", "Referral"]
PRODUCT_CATEGORIES = {
    "Electronics": ["Laptops", "Phones", "Tablets", "Accessories", "Audio", "Wearables", "Cameras"],
    "Software": ["SaaS", "On-Premise", "Mobile Apps", "Plugins", "APIs"],
    "Services": ["Consulting", "Support", "Training", "Integration", "Managed"],
    "Hardware": ["Servers", "Networking", "Storage", "Peripherals", "Components"],
    "Supplies": ["Office", "Packaging", "Raw Materials", "Maintenance", "Safety"],
}
PRODUCT_LINES = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta", "Theta"]
CURRENCIES = [
    ("USD", "US Dollar", 1.0), ("EUR", "Euro", 1.08), ("GBP", "British Pound", 1.27),
    ("JPY", "Japanese Yen", 0.0067), ("CAD", "Canadian Dollar", 0.74),
    ("AUD", "Australian Dollar", 0.65), ("CHF", "Swiss Franc", 1.12),
    ("CNY", "Chinese Yuan", 0.14), ("INR", "Indian Rupee", 0.012),
    ("KRW", "Korean Won", 0.00075), ("BRL", "Brazilian Real", 0.20),
    ("MXN", "Mexican Peso", 0.058), ("SEK", "Swedish Krona", 0.096),
    ("SGD", "Singapore Dollar", 0.74), ("HKD", "Hong Kong Dollar", 0.13),
]
WAREHOUSES = ["US-East-1", "US-West-1", "US-Central-1", "EU-West-1", "EU-Central-1", "APAC-Tokyo-1", "APAC-Sydney-1", "LATAM-SP-1"]
CARRIERS = ["FedEx", "UPS", "DHL", "USPS", "Maersk", "DB Schenker", "XPO Logistics", "Kuehne+Nagel"]
DEVICES = ["desktop", "mobile", "tablet"]
BROWSERS = ["Chrome", "Safari", "Firefox", "Edge", "Opera"]
TICKET_CATEGORIES = ["billing", "technical", "shipping", "product_quality", "account", "feature_request", "security", "integration"]
PRIORITIES = ["low", "medium", "high", "critical"]
PAYMENT_METHODS = ["credit_card", "debit_card", "wire_transfer", "ach", "paypal", "crypto", "check", "terms_net30"]
PAGES = ["/home", "/products", "/product/detail", "/cart", "/checkout", "/account", "/search", "/category", "/about", "/support"]

# Date range for the warehouse: 2020-01-01 to 2024-12-31
DATE_START = datetime(2020, 1, 1)
DATE_END = datetime(2024, 12, 31)
DATE_DAYS = (DATE_END - DATE_START).days  # 1826


def _rng_date(rng, start=None, end=None):
    s = start or DATE_START
    e = end or DATE_END
    return (s + timedelta(days=rng.randint(0, (e - s).days))).strftime("%Y-%m-%d")


def _rng_ts(rng, start=None, end=None):
    s = start or DATE_START
    e = end or DATE_END
    return (s + timedelta(seconds=rng.randint(0, int((e - s).total_seconds())))).strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------------------
# 1. Core Business Domain (dim_/fact_)  — ~25 tables
# ---------------------------------------------------------------------------

def _create_core_tables(conn, rng):
    c = conn.cursor()

    # --- Dimension tables ---
    c.execute("""CREATE TABLE dim_date (
        date_key INTEGER PRIMARY KEY, full_date TEXT NOT NULL,
        year INTEGER, quarter INTEGER, month INTEGER, month_name TEXT,
        week INTEGER, day_of_week INTEGER, day_name TEXT,
        is_weekend INTEGER, is_holiday INTEGER,
        fiscal_year INTEGER, fiscal_quarter INTEGER
    )""")
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    month_names = ["January", "February", "March", "April", "May", "June",
                   "July", "August", "September", "October", "November", "December"]
    holidays = {(1, 1), (7, 4), (12, 25), (11, 28), (9, 1)}
    for dk in range(DATE_DAYS + 1):
        d = DATE_START + timedelta(days=dk)
        wd = d.weekday()
        c.execute("INSERT INTO dim_date VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                  (dk + 1, d.strftime("%Y-%m-%d"), d.year, (d.month - 1) // 3 + 1,
                   d.month, month_names[d.month - 1], d.isocalendar()[1],
                   wd, day_names[wd], 1 if wd >= 5 else 0,
                   1 if (d.month, d.day) in holidays else 0,
                   d.year if d.month >= 7 else d.year - 1,
                   (d.month - 1) // 3 + 1))

    c.execute("""CREATE TABLE dim_customer (
        customer_key INTEGER PRIMARY KEY, customer_id TEXT UNIQUE NOT NULL,
        first_name TEXT, last_name TEXT, email TEXT, phone TEXT,
        company TEXT, industry TEXT, segment TEXT, tier TEXT,
        country TEXT, state TEXT, city TEXT, postal_code TEXT,
        signup_date TEXT, last_activity_date TEXT, is_active INTEGER DEFAULT 1,
        lifetime_value REAL DEFAULT 0, acquisition_channel TEXT,
        acquisition_campaign_id INTEGER
    )""")
    for i in range(5000):
        fn, ln = rng.choice(FIRST_NAMES), rng.choice(LAST_NAMES)
        signup = _rng_date(rng, DATE_START, datetime(2024, 6, 30))
        c.execute("INSERT INTO dim_customer VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                  (i + 1, f"CUST-{i + 1:06d}", fn, ln,
                   f"{fn.lower()}.{ln.lower()}{i}@{'example' if rng.random() > 0.3 else rng.choice(['corp', 'biz', 'co'])}.com",
                   f"+1-{rng.randint(200, 999)}-{rng.randint(100, 999)}-{rng.randint(1000, 9999)}",
                   f"{rng.choice(['Acme', 'Globe', 'Apex', 'Nova', 'Summit', 'Peak', 'Vertex'])} {rng.choice(['Inc', 'Corp', 'LLC', 'Ltd', 'Group', 'Co'])}",
                   rng.choice(INDUSTRIES), rng.choice(SEGMENTS), rng.choice(TIERS),
                   rng.choice(COUNTRIES), f"State-{rng.randint(1, 50)}", f"City-{rng.randint(1, 200)}",
                   f"{rng.randint(10000, 99999)}", signup,
                   _rng_date(rng, datetime.strptime(signup, "%Y-%m-%d"), DATE_END),
                   1 if rng.random() > 0.15 else 0,
                   round(rng.uniform(0, 50000), 2), rng.choice(CHANNELS),
                   rng.randint(1, 200) if rng.random() > 0.3 else None))

    c.execute("""CREATE TABLE dim_product (
        product_key INTEGER PRIMARY KEY, product_id TEXT UNIQUE, sku TEXT,
        name TEXT, description TEXT, category TEXT, subcategory TEXT,
        product_line TEXT, brand TEXT,
        unit_cost REAL, list_price REAL, weight_kg REAL,
        launch_date TEXT, discontinue_date TEXT,
        is_active INTEGER DEFAULT 1, supplier_key INTEGER
    )""")
    pk = 1
    for cat, subcats in PRODUCT_CATEGORIES.items():
        for subcat in subcats:
            for j in range(rng.randint(10, 20)):
                price = round(rng.uniform(15, 2000), 2)
                launch = _rng_date(rng, DATE_START, datetime(2023, 12, 31))
                disc = _rng_date(rng, datetime(2024, 1, 1), DATE_END) if rng.random() < 0.1 else None
                c.execute("INSERT INTO dim_product VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                          (pk, f"PROD-{pk:05d}", f"SKU-{cat[:2].upper()}{subcat[:2].upper()}-{pk:04d}",
                           f"{subcat} {rng.choice(['Pro', 'Plus', 'Lite', 'Max', 'Ultra', 'Basic'])} {pk}",
                           f"High-quality {subcat.lower()} product in the {cat.lower()} category.",
                           cat, subcat, rng.choice(PRODUCT_LINES),
                           rng.choice(["TechBrand", "ValueLine", "PremiumCo", "EcoGoods", "FastTrack"]),
                           round(price * rng.uniform(0.25, 0.65), 2), price,
                           round(rng.uniform(0.05, 25), 2), launch, disc,
                           0 if disc else 1, rng.randint(1, 100)))
                pk += 1
    n_products = pk - 1

    c.execute("""CREATE TABLE dim_store (
        store_key INTEGER PRIMARY KEY, store_id TEXT, store_name TEXT,
        store_type TEXT, region TEXT, country TEXT, state TEXT, city TEXT,
        open_date TEXT, close_date TEXT, square_footage INTEGER,
        manager_emp_key INTEGER
    )""")
    store_types = ["Flagship", "Standard", "Outlet", "Pop-up", "Online", "Warehouse"]
    regions = ["North America", "EMEA", "APAC", "LATAM"]
    for i in range(50):
        c.execute("INSERT INTO dim_store VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                  (i + 1, f"STORE-{i + 1:03d}", f"{rng.choice(COUNTRIES)} Store {i + 1}",
                   rng.choice(store_types), rng.choice(regions), rng.choice(COUNTRIES),
                   f"State-{rng.randint(1, 50)}", f"City-{rng.randint(1, 200)}",
                   _rng_date(rng, DATE_START, datetime(2022, 12, 31)),
                   _rng_date(rng, datetime(2024, 6, 1), DATE_END) if rng.random() < 0.05 else None,
                   rng.randint(500, 50000), rng.randint(1, 500)))

    c.execute("""CREATE TABLE dim_channel (
        channel_key INTEGER PRIMARY KEY, channel_name TEXT,
        channel_type TEXT, is_online INTEGER, parent_channel_key INTEGER
    )""")
    for i, ch in enumerate(CHANNELS):
        c.execute("INSERT INTO dim_channel VALUES (?,?,?,?,?)",
                  (i + 1, ch, rng.choice(["Direct", "Indirect", "Digital", "Physical"]),
                   1 if ch in ("Online", "Marketplace") else 0, None))

    c.execute("""CREATE TABLE dim_currency (
        currency_key INTEGER PRIMARY KEY, currency_code TEXT, currency_name TEXT,
        exchange_rate_usd REAL, last_updated TEXT
    )""")
    for i, (code, name, rate) in enumerate(CURRENCIES):
        c.execute("INSERT INTO dim_currency VALUES (?,?,?,?,?)",
                  (i + 1, code, name, rate, _rng_date(rng, datetime(2024, 1, 1), DATE_END)))

    c.execute("""CREATE TABLE dim_promotion (
        promo_key INTEGER PRIMARY KEY, promo_id TEXT, promo_name TEXT,
        promo_type TEXT, discount_pct REAL, start_date TEXT, end_date TEXT,
        min_purchase REAL, channel_key INTEGER
    )""")
    promo_types = ["Percentage", "BOGO", "Free Shipping", "Bundle", "Loyalty", "Seasonal", "Flash Sale"]
    for i in range(100):
        start = _rng_date(rng)
        c.execute("INSERT INTO dim_promotion VALUES (?,?,?,?,?,?,?,?,?)",
                  (i + 1, f"PROMO-{i + 1:04d}", f"{rng.choice(promo_types)} {rng.choice(['Summer', 'Winter', 'Spring', 'Fall', 'Holiday', 'Launch'])} {i + 1}",
                   rng.choice(promo_types), round(rng.uniform(5, 50), 1), start,
                   _rng_date(rng, datetime.strptime(start, "%Y-%m-%d"), DATE_END),
                   round(rng.uniform(0, 500), 2), rng.randint(1, len(CHANNELS))))

    c.execute("""CREATE TABLE dim_geography (
        geo_key INTEGER PRIMARY KEY, country TEXT, country_code TEXT,
        region TEXT, subregion TEXT, state TEXT, city TEXT,
        timezone TEXT, population INTEGER
    )""")
    for i in range(200):
        country = rng.choice(COUNTRIES)
        c.execute("INSERT INTO dim_geography VALUES (?,?,?,?,?,?,?,?,?)",
                  (i + 1, country, country, rng.choice(regions),
                   f"Sub-{rng.randint(1, 10)}", f"State-{rng.randint(1, 50)}",
                   f"City-{rng.randint(1, 200)}", f"UTC{rng.choice(['-5', '-8', '+0', '+1', '+9', '+10'])}",
                   rng.randint(10000, 20000000)))

    # --- Fact tables ---
    c.execute("""CREATE TABLE fact_sales (
        sale_id INTEGER PRIMARY KEY, customer_key INTEGER, product_key INTEGER,
        store_key INTEGER, date_key INTEGER, channel_key INTEGER,
        promo_key INTEGER, currency_key INTEGER,
        quantity INTEGER, unit_price REAL, discount_amount REAL,
        tax_amount REAL, shipping_amount REAL,
        revenue REAL, cost_of_goods REAL, profit REAL,
        order_id TEXT, order_date TEXT, ship_date TEXT, delivery_date TEXT,
        status TEXT,
        FOREIGN KEY (customer_key) REFERENCES dim_customer(customer_key),
        FOREIGN KEY (product_key) REFERENCES dim_product(product_key)
    )""")
    statuses = ["completed", "shipped", "processing", "cancelled", "returned", "refunded"]
    for i in range(50000):
        odate = DATE_START + timedelta(days=rng.randint(0, DATE_DAYS))
        date_key = (odate - DATE_START).days + 1
        qty = rng.randint(1, 20)
        price = round(rng.uniform(10, 1500), 2)
        disc = round(price * qty * rng.uniform(0, 0.3), 2) if rng.random() > 0.5 else 0
        revenue = round(price * qty - disc, 2)
        cogs = round(revenue * rng.uniform(0.3, 0.7), 2)
        c.execute("INSERT INTO fact_sales VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                  (i + 1, rng.randint(1, 5000), rng.randint(1, n_products),
                   rng.randint(1, 50), date_key, rng.randint(1, len(CHANNELS)),
                   rng.randint(1, 100) if rng.random() > 0.6 else None,
                   rng.randint(1, len(CURRENCIES)),
                   qty, price, disc,
                   round(revenue * rng.uniform(0.05, 0.12), 2),
                   round(rng.uniform(0, 25), 2) if rng.random() > 0.3 else 0,
                   revenue, cogs, round(revenue - cogs, 2),
                   f"ORD-{i // rng.randint(1, 5) + 1:07d}", odate.strftime("%Y-%m-%d"),
                   (odate + timedelta(days=rng.randint(0, 5))).strftime("%Y-%m-%d"),
                   (odate + timedelta(days=rng.randint(3, 14))).strftime("%Y-%m-%d"),
                   rng.choice(statuses)))

    c.execute("""CREATE TABLE fact_returns (
        return_id INTEGER PRIMARY KEY, sale_id INTEGER, customer_key INTEGER,
        product_key INTEGER, date_key INTEGER,
        reason_code TEXT, refund_amount REAL, return_date TEXT,
        resolution TEXT, restocking_fee REAL,
        FOREIGN KEY (sale_id) REFERENCES fact_sales(sale_id)
    )""")
    reasons = ["defective", "wrong_item", "not_as_described", "changed_mind", "late_delivery", "damaged_shipping", "duplicate_order"]
    for i in range(5000):
        rdate = _rng_date(rng, datetime(2021, 1, 1), DATE_END)
        c.execute("INSERT INTO fact_returns VALUES (?,?,?,?,?,?,?,?,?,?)",
                  (i + 1, rng.randint(1, 50000), rng.randint(1, 5000),
                   rng.randint(1, n_products),
                   (datetime.strptime(rdate, "%Y-%m-%d") - DATE_START).days + 1,
                   rng.choice(reasons), round(rng.uniform(10, 1500), 2), rdate,
                   rng.choice(["refunded", "replaced", "store_credit", "rejected", "pending"]),
                   round(rng.uniform(0, 50), 2)))

    c.execute("""CREATE TABLE fact_inventory_snapshot (
        snapshot_id INTEGER PRIMARY KEY, product_key INTEGER, store_key INTEGER,
        date_key INTEGER, quantity_on_hand INTEGER, quantity_reserved INTEGER,
        quantity_available INTEGER, reorder_point INTEGER, days_of_supply REAL
    )""")
    sid = 1
    for prod in range(1, min(n_products + 1, 201)):
        for store in rng.sample(range(1, 51), min(5, 50)):
            qoh = rng.randint(0, 500)
            qr = rng.randint(0, min(qoh, 50))
            c.execute("INSERT INTO fact_inventory_snapshot VALUES (?,?,?,?,?,?,?,?,?)",
                      (sid, prod, store, rng.randint(1, DATE_DAYS),
                       qoh, qr, qoh - qr, rng.randint(10, 100),
                       round(rng.uniform(0, 90), 1)))
            sid += 1

    c.execute("""CREATE TABLE fact_web_traffic (
        event_id INTEGER PRIMARY KEY, customer_key INTEGER, session_id TEXT,
        date_key INTEGER, event_type TEXT, page_url TEXT,
        referrer_url TEXT, device_type TEXT, browser TEXT,
        duration_seconds INTEGER, is_conversion INTEGER
    )""")
    event_types = ["page_view", "click", "add_to_cart", "remove_from_cart", "checkout_start", "purchase", "search", "signup", "login"]

    for i in range(30000):
        c.execute("INSERT INTO fact_web_traffic VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                  (i + 1, rng.randint(1, 5000) if rng.random() > 0.3 else None,
                   f"SESS-{rng.randint(1, 50000):08d}", rng.randint(1, DATE_DAYS),
                   rng.choice(event_types), rng.choice(PAGES) + f"/{rng.randint(1, 500)}",
                   rng.choice(["google.com", "facebook.com", "direct", "email", "partner.com", ""]),
                   rng.choice(DEVICES), rng.choice(BROWSERS),
                   rng.randint(0, 600), 1 if rng.random() < 0.05 else 0))

    # Aggregation tables
    c.execute("""CREATE TABLE fact_daily_sales_agg (
        agg_id INTEGER PRIMARY KEY, date_key INTEGER, product_key INTEGER,
        store_key INTEGER, channel_key INTEGER,
        total_quantity INTEGER, total_revenue REAL, total_cost REAL,
        total_discount REAL, order_count INTEGER
    )""")
    c.execute("""CREATE TABLE fact_monthly_customer_agg (
        agg_id INTEGER PRIMARY KEY, customer_key INTEGER,
        year INTEGER, month INTEGER,
        order_count INTEGER, total_revenue REAL, total_items INTEGER,
        avg_order_value REAL, return_count INTEGER
    )""")
    c.execute("""CREATE TABLE fact_product_reviews (
        review_id INTEGER PRIMARY KEY, product_key INTEGER, customer_key INTEGER,
        review_date TEXT, rating INTEGER, title TEXT, body TEXT,
        helpful_votes INTEGER, verified_purchase INTEGER
    )""")
    for i in range(8000):
        c.execute("INSERT INTO fact_product_reviews VALUES (?,?,?,?,?,?,?,?,?)",
                  (i + 1, rng.randint(1, n_products), rng.randint(1, 5000),
                   _rng_date(rng), rng.randint(1, 5),
                   f"Review title {i}", f"Review body for product review {i}.",
                   rng.randint(0, 100), rng.randint(0, 1)))

    return n_products


# ---------------------------------------------------------------------------
# 2. Finance Domain (fin_)  — ~15 tables
# ---------------------------------------------------------------------------

def _create_finance_tables(conn, rng):
    c = conn.cursor()

    c.execute("""CREATE TABLE fin_chart_of_accounts (
        account_id INTEGER PRIMARY KEY, account_number TEXT UNIQUE,
        account_name TEXT, account_type TEXT, parent_account_id INTEGER,
        is_active INTEGER DEFAULT 1, created_at TEXT
    )""")
    acct_types = ["Asset", "Liability", "Equity", "Revenue", "Expense"]
    for i in range(200):
        c.execute("INSERT INTO fin_chart_of_accounts VALUES (?,?,?,?,?,?,?)",
                  (i + 1, f"{rng.randint(1000, 9999)}-{rng.randint(100, 999)}",
                   f"Account {i + 1} - {rng.choice(acct_types)}", rng.choice(acct_types),
                   rng.randint(1, max(1, i)) if i > 10 else None,
                   1 if rng.random() > 0.1 else 0, _rng_date(rng, DATE_START, datetime(2021, 12, 31))))

    c.execute("""CREATE TABLE fin_journal_entries (
        entry_id INTEGER PRIMARY KEY, journal_id TEXT, account_id INTEGER,
        posting_date TEXT, amount REAL, debit_credit TEXT,
        description TEXT, created_by INTEGER, approved_by INTEGER,
        FOREIGN KEY (account_id) REFERENCES fin_chart_of_accounts(account_id)
    )""")
    for i in range(20000):
        c.execute("INSERT INTO fin_journal_entries VALUES (?,?,?,?,?,?,?,?,?)",
                  (i + 1, f"JE-{i // 2 + 1:06d}", rng.randint(1, 200),
                   _rng_date(rng), round(rng.uniform(10, 100000), 2),
                   rng.choice(["D", "C"]), f"Journal entry {i + 1}",
                   rng.randint(1, 500), rng.randint(1, 500) if rng.random() > 0.2 else None))

    c.execute("""CREATE TABLE fin_general_ledger (
        gl_id INTEGER PRIMARY KEY, account_id INTEGER, period TEXT,
        fiscal_year INTEGER, beginning_balance REAL,
        debits REAL, credits REAL, ending_balance REAL
    )""")
    gl_id = 1
    for acct in range(1, 51):
        for yr in range(2020, 2025):
            for mo in range(1, 13):
                bb = round(rng.uniform(-100000, 500000), 2)
                db = round(rng.uniform(0, 200000), 2)
                cr = round(rng.uniform(0, 200000), 2)
                c.execute("INSERT INTO fin_general_ledger VALUES (?,?,?,?,?,?,?,?)",
                          (gl_id, acct, f"{yr}-{mo:02d}", yr, bb, db, cr, round(bb + db - cr, 2)))
                gl_id += 1

    c.execute("""CREATE TABLE fin_invoices (
        invoice_id INTEGER PRIMARY KEY, customer_key INTEGER,
        invoice_date TEXT, due_date TEXT, total_amount REAL,
        tax_amount REAL, status TEXT, payment_terms TEXT, currency_key INTEGER
    )""")
    inv_statuses = ["paid", "unpaid", "overdue", "partially_paid", "voided", "disputed"]
    for i in range(15000):
        idate = _rng_date(rng)
        c.execute("INSERT INTO fin_invoices VALUES (?,?,?,?,?,?,?,?,?)",
                  (i + 1, rng.randint(1, 5000), idate,
                   (datetime.strptime(idate, "%Y-%m-%d") + timedelta(days=rng.choice([15, 30, 45, 60, 90]))).strftime("%Y-%m-%d"),
                   round(rng.uniform(50, 50000), 2), round(rng.uniform(5, 5000), 2),
                   rng.choice(inv_statuses), rng.choice(["Net 15", "Net 30", "Net 45", "Net 60", "Net 90"]),
                   rng.randint(1, len(CURRENCIES))))

    c.execute("""CREATE TABLE fin_invoice_lines (
        line_id INTEGER PRIMARY KEY, invoice_id INTEGER,
        product_key INTEGER, quantity INTEGER,
        unit_price REAL, line_total REAL, tax_rate REAL
    )""")
    lid = 1
    for inv in range(1, 15001):
        for _ in range(rng.randint(1, 4)):
            qty = rng.randint(1, 50)
            price = round(rng.uniform(10, 2000), 2)
            c.execute("INSERT INTO fin_invoice_lines VALUES (?,?,?,?,?,?,?)",
                      (lid, inv, rng.randint(1, 500), qty, price,
                       round(qty * price, 2), round(rng.uniform(0.05, 0.15), 3)))
            lid += 1

    c.execute("""CREATE TABLE fin_payments (
        payment_id INTEGER PRIMARY KEY, invoice_id INTEGER,
        payment_date TEXT, amount REAL, payment_method TEXT,
        reference_number TEXT, status TEXT
    )""")
    for i in range(12000):
        c.execute("INSERT INTO fin_payments VALUES (?,?,?,?,?,?,?)",
                  (i + 1, rng.randint(1, 15000), _rng_date(rng),
                   round(rng.uniform(50, 50000), 2), rng.choice(PAYMENT_METHODS),
                   f"REF-{rng.randint(100000, 999999)}", rng.choice(["completed", "pending", "failed", "reversed"])))

    c.execute("""CREATE TABLE fin_budgets (
        budget_id INTEGER PRIMARY KEY, department_key INTEGER, account_id INTEGER,
        fiscal_year INTEGER, fiscal_quarter INTEGER,
        budget_amount REAL, actual_amount REAL, variance REAL
    )""")
    bid = 1
    for dept in range(1, 31):
        for yr in range(2020, 2025):
            for q in range(1, 5):
                ba = round(rng.uniform(50000, 2000000), 2)
                aa = round(ba * rng.uniform(0.7, 1.4), 2)
                c.execute("INSERT INTO fin_budgets VALUES (?,?,?,?,?,?,?,?)",
                          (bid, dept, rng.randint(1, 200), yr, q, ba, aa, round(aa - ba, 2)))
                bid += 1

    c.execute("""CREATE TABLE fin_cost_centers (
        cost_center_id INTEGER PRIMARY KEY, name TEXT,
        department_key INTEGER, manager_emp_key INTEGER, budget_allocated REAL
    )""")
    for i in range(60):
        c.execute("INSERT INTO fin_cost_centers VALUES (?,?,?,?,?)",
                  (i + 1, f"CC-{rng.choice(['Engineering', 'Sales', 'Marketing', 'Ops', 'Support', 'R&D'])}-{i + 1}",
                   rng.randint(1, 30), rng.randint(1, 500), round(rng.uniform(100000, 5000000), 2)))

    c.execute("""CREATE TABLE fin_accounts_receivable (
        ar_id INTEGER PRIMARY KEY, customer_key INTEGER, invoice_id INTEGER,
        amount_due REAL, amount_paid REAL, days_outstanding INTEGER,
        aging_bucket TEXT
    )""")
    buckets = ["Current", "1-30 days", "31-60 days", "61-90 days", "90+ days"]
    for i in range(8000):
        days = rng.randint(0, 365)
        c.execute("INSERT INTO fin_accounts_receivable VALUES (?,?,?,?,?,?,?)",
                  (i + 1, rng.randint(1, 5000), rng.randint(1, 15000),
                   round(rng.uniform(100, 50000), 2), round(rng.uniform(0, 50000), 2),
                   days, buckets[min(days // 30, 4)]))

    c.execute("""CREATE TABLE fin_accounts_payable (
        ap_id INTEGER PRIMARY KEY, supplier_key INTEGER, po_id INTEGER,
        amount_due REAL, amount_paid REAL, due_date TEXT, status TEXT
    )""")
    for i in range(6000):
        c.execute("INSERT INTO fin_accounts_payable VALUES (?,?,?,?,?,?,?)",
                  (i + 1, rng.randint(1, 100), rng.randint(1, 5000),
                   round(rng.uniform(500, 200000), 2), round(rng.uniform(0, 200000), 2),
                   _rng_date(rng), rng.choice(["paid", "unpaid", "overdue", "partial"])))

    c.execute("""CREATE TABLE fin_tax_rates (
        tax_id INTEGER PRIMARY KEY, jurisdiction TEXT, tax_type TEXT,
        rate REAL, effective_date TEXT, end_date TEXT
    )""")
    for i in range(50):
        c.execute("INSERT INTO fin_tax_rates VALUES (?,?,?,?,?,?)",
                  (i + 1, rng.choice(COUNTRIES), rng.choice(["sales_tax", "vat", "gst", "excise"]),
                   round(rng.uniform(0.01, 0.25), 4),
                   _rng_date(rng, DATE_START, datetime(2022, 12, 31)), None))

    c.execute("""CREATE TABLE fin_exchange_rates (
        rate_id INTEGER PRIMARY KEY, from_currency TEXT, to_currency TEXT,
        rate REAL, effective_date TEXT
    )""")
    rid = 1
    for _ in range(500):
        c.execute("INSERT INTO fin_exchange_rates VALUES (?,?,?,?,?)",
                  (rid, rng.choice([c[0] for c in CURRENCIES]),
                   rng.choice([c[0] for c in CURRENCIES]),
                   round(rng.uniform(0.001, 150), 6), _rng_date(rng)))
        rid += 1


# ---------------------------------------------------------------------------
# 3. HR Domain (hr_)  — ~12 tables
# ---------------------------------------------------------------------------

def _create_hr_tables(conn, rng):
    c = conn.cursor()

    c.execute("""CREATE TABLE hr_departments (
        department_key INTEGER PRIMARY KEY, dept_id TEXT,
        department_name TEXT, parent_dept_key INTEGER,
        vp_emp_key INTEGER, budget REAL,
        cost_center_id INTEGER, location TEXT
    )""")
    dept_names = [
        "Engineering", "Sales", "Marketing", "Finance", "HR", "Operations",
        "Legal", "Support", "Product", "Data Science", "IT", "Security",
        "Research", "QA", "DevOps", "Design", "Analytics", "Partnerships",
        "Customer Success", "Compliance", "Procurement", "Facilities",
        "Corporate Development", "Investor Relations", "Communications",
        "Internal Audit", "Risk Management", "Strategy", "Innovation Lab", "Platform",
    ]
    for i, dn in enumerate(dept_names):
        c.execute("INSERT INTO hr_departments VALUES (?,?,?,?,?,?,?,?)",
                  (i + 1, f"DEPT-{i + 1:03d}", dn,
                   rng.randint(1, max(1, i)) if i > 5 else None,
                   rng.randint(1, 50), round(rng.uniform(500000, 20000000), 2),
                   rng.randint(1, 60), rng.choice(["NYC", "SF", "Austin", "London", "Berlin", "Tokyo", "Remote"])))

    c.execute("""CREATE TABLE hr_positions (
        position_key INTEGER PRIMARY KEY, position_title TEXT,
        job_family TEXT, job_level INTEGER,
        min_salary REAL, max_salary REAL, is_exempt INTEGER
    )""")
    families = ["Engineering", "Sales", "Marketing", "Finance", "Operations", "Management", "Support", "Research", "Design", "Data"]
    titles = ["Intern", "Associate", "Specialist", "Senior Specialist", "Lead", "Manager", "Senior Manager", "Director", "VP", "SVP", "C-Level"]
    pk = 1
    for fam in families:
        for lvl, title in enumerate(titles):
            min_s = 40000 + lvl * 20000
            c.execute("INSERT INTO hr_positions VALUES (?,?,?,?,?,?,?)",
                      (pk, f"{title} - {fam}", fam, lvl + 1,
                       min_s, min_s + 40000 + lvl * 15000,
                       0 if lvl < 2 else 1))
            pk += 1

    c.execute("""CREATE TABLE hr_employees (
        emp_key INTEGER PRIMARY KEY, employee_id TEXT UNIQUE,
        first_name TEXT, last_name TEXT, email TEXT, phone TEXT,
        department_key INTEGER, position_key INTEGER, manager_key INTEGER,
        hire_date TEXT, termination_date TEXT, employment_status TEXT,
        salary REAL, bonus_target_pct REAL, location TEXT, is_remote INTEGER,
        FOREIGN KEY (department_key) REFERENCES hr_departments(department_key)
    )""")
    emp_statuses = ["active", "terminated", "on_leave", "contractor"]
    for i in range(1000):
        fn, ln = rng.choice(FIRST_NAMES), rng.choice(LAST_NAMES)
        hire = _rng_date(rng, datetime(2016, 1, 1), datetime(2024, 6, 30))
        term = _rng_date(rng, datetime.strptime(hire, "%Y-%m-%d"), DATE_END) if rng.random() < 0.15 else None
        c.execute("INSERT INTO hr_employees VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                  (i + 1, f"EMP-{i + 1:05d}", fn, ln,
                   f"{fn.lower()}.{ln.lower()}@company.com",
                   f"+1-{rng.randint(200, 999)}-{rng.randint(100, 999)}-{rng.randint(1000, 9999)}",
                   rng.randint(1, 30), rng.randint(1, pk - 1),
                   rng.randint(1, max(1, i)) if i > 0 else None,
                   hire, term,
                   "terminated" if term else rng.choice(["active"] * 9 + ["on_leave", "contractor"]),
                   round(rng.uniform(45000, 350000), 2),
                   round(rng.uniform(0, 0.4), 2),
                   rng.choice(["NYC", "SF", "Austin", "London", "Berlin", "Tokyo", "Remote"]),
                   1 if rng.random() > 0.6 else 0))

    c.execute("""CREATE TABLE hr_payroll (
        payroll_id INTEGER PRIMARY KEY, emp_key INTEGER,
        pay_period_start TEXT, pay_period_end TEXT,
        gross_pay REAL, deductions REAL, net_pay REAL,
        hours_worked REAL, overtime_hours REAL
    )""")
    pid = 1
    for emp in range(1, 201):
        for mo in range(rng.randint(6, 24)):
            gp = round(rng.uniform(3000, 25000), 2)
            ded = round(gp * rng.uniform(0.15, 0.35), 2)
            c.execute("INSERT INTO hr_payroll VALUES (?,?,?,?,?,?,?,?,?)",
                      (pid, emp,
                       (datetime(2023, 1, 1) + timedelta(days=mo * 14)).strftime("%Y-%m-%d"),
                       (datetime(2023, 1, 1) + timedelta(days=mo * 14 + 13)).strftime("%Y-%m-%d"),
                       gp, ded, round(gp - ded, 2),
                       round(rng.uniform(70, 100), 1), round(rng.uniform(0, 20), 1)))
            pid += 1

    c.execute("""CREATE TABLE hr_performance_reviews (
        review_id INTEGER PRIMARY KEY, emp_key INTEGER, reviewer_key INTEGER,
        review_period TEXT, overall_rating REAL,
        goal_completion_pct REAL, review_date TEXT, comments TEXT
    )""")
    for i in range(2000):
        c.execute("INSERT INTO hr_performance_reviews VALUES (?,?,?,?,?,?,?,?)",
                  (i + 1, rng.randint(1, 1000), rng.randint(1, 1000),
                   f"{rng.choice([2022, 2023, 2024])}-H{rng.choice([1, 2])}",
                   round(rng.uniform(1, 5), 1), round(rng.uniform(30, 100), 1),
                   _rng_date(rng), f"Performance review comments for employee review {i + 1}."))

    c.execute("""CREATE TABLE hr_compensation_history (
        comp_id INTEGER PRIMARY KEY, emp_key INTEGER, effective_date TEXT,
        salary REAL, bonus_pct REAL, equity_shares INTEGER, change_reason TEXT
    )""")
    for i in range(3000):
        c.execute("INSERT INTO hr_compensation_history VALUES (?,?,?,?,?,?,?)",
                  (i + 1, rng.randint(1, 1000), _rng_date(rng),
                   round(rng.uniform(45000, 350000), 2), round(rng.uniform(0, 0.4), 2),
                   rng.randint(0, 10000),
                   rng.choice(["annual_review", "promotion", "market_adjustment", "new_hire", "role_change"])))

    c.execute("""CREATE TABLE hr_benefits_enrollment (
        enrollment_id INTEGER PRIMARY KEY, emp_key INTEGER,
        benefit_plan_id INTEGER, coverage_level TEXT,
        employee_cost REAL, employer_cost REAL,
        effective_date TEXT, end_date TEXT
    )""")
    for i in range(2000):
        c.execute("INSERT INTO hr_benefits_enrollment VALUES (?,?,?,?,?,?,?,?)",
                  (i + 1, rng.randint(1, 1000), rng.randint(1, 20),
                   rng.choice(["employee_only", "employee_spouse", "family"]),
                   round(rng.uniform(50, 800), 2), round(rng.uniform(200, 2000), 2),
                   _rng_date(rng), None))

    c.execute("""CREATE TABLE hr_benefit_plans (
        plan_id INTEGER PRIMARY KEY, plan_name TEXT, plan_type TEXT,
        carrier TEXT, annual_cost REAL
    )""")
    for i in range(20):
        c.execute("INSERT INTO hr_benefit_plans VALUES (?,?,?,?,?)",
                  (i + 1, f"{rng.choice(['Health', 'Dental', 'Vision', 'Life', '401k', 'HSA', 'FSA', 'Disability'])} Plan {rng.choice(['A', 'B', 'C'])}",
                   rng.choice(["health", "dental", "vision", "life", "retirement", "savings"]),
                   rng.choice(["Aetna", "UHC", "Kaiser", "Cigna", "BCBS", "MetLife", "Fidelity"]),
                   round(rng.uniform(1000, 20000), 2)))

    c.execute("""CREATE TABLE hr_time_off (
        time_off_id INTEGER PRIMARY KEY, emp_key INTEGER,
        leave_type TEXT, start_date TEXT, end_date TEXT,
        hours REAL, status TEXT, approved_by INTEGER
    )""")
    for i in range(5000):
        sd = _rng_date(rng)
        c.execute("INSERT INTO hr_time_off VALUES (?,?,?,?,?,?,?,?)",
                  (i + 1, rng.randint(1, 1000),
                   rng.choice(["vacation", "sick", "personal", "bereavement", "jury_duty", "parental"]),
                   sd, (datetime.strptime(sd, "%Y-%m-%d") + timedelta(days=rng.randint(1, 14))).strftime("%Y-%m-%d"),
                   rng.choice([8, 16, 24, 40, 80]),
                   rng.choice(["approved", "pending", "denied"]),
                   rng.randint(1, 1000)))

    c.execute("""CREATE TABLE hr_training (
        training_id INTEGER PRIMARY KEY, emp_key INTEGER,
        course_name TEXT, course_type TEXT,
        completion_date TEXT, score REAL, is_mandatory INTEGER
    )""")
    courses = ["Security Awareness", "Leadership 101", "Python Advanced", "Data Privacy", "Project Management",
               "SQL Mastery", "Cloud Architecture", "Agile Methodology", "Communication Skills", "Ethics Training"]
    for i in range(3000):
        c.execute("INSERT INTO hr_training VALUES (?,?,?,?,?,?,?)",
                  (i + 1, rng.randint(1, 1000), rng.choice(courses),
                   rng.choice(["online", "classroom", "workshop", "certification"]),
                   _rng_date(rng), round(rng.uniform(50, 100), 1),
                   1 if rng.random() > 0.6 else 0))


# ---------------------------------------------------------------------------
# 4. Marketing Domain (mktg_)  — ~10 tables
# ---------------------------------------------------------------------------

def _create_marketing_tables(conn, rng):
    c = conn.cursor()

    c.execute("""CREATE TABLE mktg_campaigns (
        campaign_id INTEGER PRIMARY KEY, campaign_name TEXT,
        campaign_type TEXT, channel_key INTEGER,
        start_date TEXT, end_date TEXT, budget REAL, spend REAL,
        status TEXT, objective TEXT, target_segment TEXT
    )""")
    camp_types = ["Email", "SEM", "Social", "Display", "Content", "Event", "Webinar", "Partner", "Direct Mail", "TV"]
    for i in range(200):
        sd = _rng_date(rng)
        c.execute("INSERT INTO mktg_campaigns VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                  (i + 1, f"{rng.choice(camp_types)} Campaign {rng.choice(['Q1', 'Q2', 'Q3', 'Q4'])} {rng.choice([2022, 2023, 2024])} #{i + 1}",
                   rng.choice(camp_types), rng.randint(1, len(CHANNELS)),
                   sd, (datetime.strptime(sd, "%Y-%m-%d") + timedelta(days=rng.randint(7, 90))).strftime("%Y-%m-%d"),
                   round(rng.uniform(5000, 500000), 2), round(rng.uniform(1000, 500000), 2),
                   rng.choice(["active", "completed", "paused", "draft"]),
                   rng.choice(["awareness", "lead_gen", "conversion", "retention", "upsell"]),
                   rng.choice(SEGMENTS)))

    c.execute("""CREATE TABLE mktg_leads (
        lead_id INTEGER PRIMARY KEY, first_name TEXT, last_name TEXT,
        email TEXT, company TEXT, source TEXT, campaign_id INTEGER,
        created_date TEXT, status TEXT, score INTEGER,
        assigned_to INTEGER, converted_customer_key INTEGER
    )""")
    for i in range(10000):
        c.execute("INSERT INTO mktg_leads VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                  (i + 1, rng.choice(FIRST_NAMES), rng.choice(LAST_NAMES),
                   f"lead{i}@{rng.choice(['gmail', 'outlook', 'company', 'corp'])}.com",
                   f"{rng.choice(['Tech', 'Global', 'Smart', 'First'])} {rng.choice(['Corp', 'Inc', 'LLC'])}",
                   rng.choice(["organic", "paid_search", "social", "referral", "event", "cold_outreach"]),
                   rng.randint(1, 200), _rng_date(rng),
                   rng.choice(["new", "contacted", "qualified", "converted", "lost", "nurturing"]),
                   rng.randint(1, 100), rng.randint(1, 200),
                   rng.randint(1, 5000) if rng.random() < 0.15 else None))

    c.execute("""CREATE TABLE mktg_attribution (
        attribution_id INTEGER PRIMARY KEY, customer_key INTEGER,
        campaign_id INTEGER, channel_key INTEGER,
        touchpoint_date TEXT, touchpoint_type TEXT,
        attribution_model TEXT, revenue_attributed REAL
    )""")
    models = ["first_touch", "last_touch", "linear", "time_decay", "position_based"]
    for i in range(20000):
        c.execute("INSERT INTO mktg_attribution VALUES (?,?,?,?,?,?,?,?)",
                  (i + 1, rng.randint(1, 5000), rng.randint(1, 200),
                   rng.randint(1, len(CHANNELS)), _rng_date(rng),
                   rng.choice(["impression", "click", "form_fill", "demo_request", "purchase"]),
                   rng.choice(models), round(rng.uniform(0, 5000), 2)))

    c.execute("""CREATE TABLE mktg_segments (
        segment_id INTEGER PRIMARY KEY, segment_name TEXT,
        segment_type TEXT, criteria TEXT,
        customer_count INTEGER, created_date TEXT, last_refreshed TEXT
    )""")
    for i in range(50):
        c.execute("INSERT INTO mktg_segments VALUES (?,?,?,?,?,?,?)",
                  (i + 1, f"Segment {rng.choice(['High Value', 'At Risk', 'New', 'Power', 'Dormant', 'Loyal', 'Price Sensitive'])} {i + 1}",
                   rng.choice(["behavioral", "demographic", "firmographic", "predictive"]),
                   f"tier IN ('{rng.choice(TIERS)}') AND country = '{rng.choice(COUNTRIES)}'",
                   rng.randint(50, 2000), _rng_date(rng), _rng_date(rng)))

    c.execute("""CREATE TABLE mktg_email_sends (
        send_id INTEGER PRIMARY KEY, campaign_id INTEGER,
        recipient_email TEXT, sent_at TEXT,
        delivered INTEGER, opened INTEGER, clicked INTEGER,
        unsubscribed INTEGER, bounced INTEGER
    )""")
    for i in range(30000):
        c.execute("INSERT INTO mktg_email_sends VALUES (?,?,?,?,?,?,?,?,?)",
                  (i + 1, rng.randint(1, 200),
                   f"user{rng.randint(1, 10000)}@example.com", _rng_ts(rng),
                   1 if rng.random() > 0.05 else 0,
                   1 if rng.random() > 0.7 else 0,
                   1 if rng.random() > 0.9 else 0,
                   1 if rng.random() < 0.02 else 0,
                   1 if rng.random() < 0.05 else 0))

    c.execute("""CREATE TABLE mktg_ab_tests (
        test_id INTEGER PRIMARY KEY, campaign_id INTEGER,
        test_name TEXT, variant TEXT, metric TEXT,
        value REAL, sample_size INTEGER, confidence REAL,
        start_date TEXT, end_date TEXT, is_winner INTEGER
    )""")
    for i in range(100):
        c.execute("INSERT INTO mktg_ab_tests VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                  (i + 1, rng.randint(1, 200), f"Test {i + 1}",
                   rng.choice(["A", "B", "C"]),
                   rng.choice(["ctr", "conversion_rate", "revenue_per_user", "open_rate"]),
                   round(rng.uniform(0, 0.3), 4), rng.randint(1000, 50000),
                   round(rng.uniform(0.5, 0.99), 3),
                   _rng_date(rng), _rng_date(rng), rng.randint(0, 1)))

    c.execute("""CREATE TABLE mktg_content (
        content_id INTEGER PRIMARY KEY, title TEXT, content_type TEXT,
        campaign_id INTEGER, author TEXT, publish_date TEXT,
        pageviews INTEGER, avg_time_on_page REAL, bounce_rate REAL
    )""")
    for i in range(500):
        c.execute("INSERT INTO mktg_content VALUES (?,?,?,?,?,?,?,?,?)",
                  (i + 1, f"Content piece {i + 1}: {rng.choice(['How to', 'Guide to', 'Best practices for', 'Introduction to', 'Deep dive into'])} {rng.choice(['analytics', 'marketing', 'sales', 'growth', 'AI'])}",
                   rng.choice(["blog", "whitepaper", "case_study", "video", "infographic", "webinar"]),
                   rng.randint(1, 200), f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}",
                   _rng_date(rng), rng.randint(100, 50000),
                   round(rng.uniform(30, 600), 1), round(rng.uniform(0.2, 0.9), 3)))

    c.execute("""CREATE TABLE mktg_social_metrics (
        metric_id INTEGER PRIMARY KEY, platform TEXT, post_id TEXT,
        campaign_id INTEGER, date TEXT, impressions INTEGER,
        engagements INTEGER, clicks INTEGER, shares INTEGER, sentiment_score REAL
    )""")
    platforms = ["Twitter", "LinkedIn", "Facebook", "Instagram", "TikTok", "YouTube"]
    for i in range(5000):
        c.execute("INSERT INTO mktg_social_metrics VALUES (?,?,?,?,?,?,?,?,?,?)",
                  (i + 1, rng.choice(platforms), f"POST-{rng.randint(1, 10000):06d}",
                   rng.randint(1, 200), _rng_date(rng),
                   rng.randint(100, 1000000), rng.randint(0, 50000),
                   rng.randint(0, 10000), rng.randint(0, 5000),
                   round(rng.uniform(-1, 1), 3)))


# ---------------------------------------------------------------------------
# 5. Supply Chain Domain (sc_)  — ~10 tables
# ---------------------------------------------------------------------------

def _create_supply_chain_tables(conn, rng):
    c = conn.cursor()

    c.execute("""CREATE TABLE sc_suppliers (
        supplier_key INTEGER PRIMARY KEY, supplier_id TEXT UNIQUE,
        supplier_name TEXT, country TEXT, contact_email TEXT,
        rating REAL, contract_start TEXT, contract_end TEXT,
        payment_terms TEXT, is_preferred INTEGER
    )""")
    for i in range(100):
        cs = _rng_date(rng, DATE_START, datetime(2023, 1, 1))
        c.execute("INSERT INTO sc_suppliers VALUES (?,?,?,?,?,?,?,?,?,?)",
                  (i + 1, f"SUP-{i + 1:04d}",
                   f"{rng.choice(['Global', 'Premier', 'Pacific', 'Atlantic', 'Central'])} {rng.choice(['Supply', 'Parts', 'Materials', 'Components', 'Solutions'])} {rng.choice(['Co', 'Inc', 'Ltd', 'Corp', 'GmbH'])}",
                   rng.choice(COUNTRIES), f"sales@supplier{i + 1}.com",
                   round(rng.uniform(1, 5), 1), cs,
                   (datetime.strptime(cs, "%Y-%m-%d") + timedelta(days=rng.randint(365, 1095))).strftime("%Y-%m-%d"),
                   rng.choice(["Net 30", "Net 45", "Net 60", "2/10 Net 30"]),
                   1 if rng.random() > 0.7 else 0))

    c.execute("""CREATE TABLE sc_purchase_orders (
        po_id INTEGER PRIMARY KEY, supplier_key INTEGER,
        order_date TEXT, expected_delivery TEXT, actual_delivery TEXT,
        total_amount REAL, status TEXT, buyer_emp_key INTEGER
    )""")
    for i in range(5000):
        od = _rng_date(rng)
        exp = (datetime.strptime(od, "%Y-%m-%d") + timedelta(days=rng.randint(7, 60))).strftime("%Y-%m-%d")
        c.execute("INSERT INTO sc_purchase_orders VALUES (?,?,?,?,?,?,?,?)",
                  (i + 1, rng.randint(1, 100), od, exp,
                   (datetime.strptime(exp, "%Y-%m-%d") + timedelta(days=rng.randint(-5, 15))).strftime("%Y-%m-%d") if rng.random() > 0.1 else None,
                   round(rng.uniform(500, 500000), 2),
                   rng.choice(["pending", "approved", "shipped", "received", "cancelled"]),
                   rng.randint(1, 500)))

    c.execute("""CREATE TABLE sc_po_lines (
        po_line_id INTEGER PRIMARY KEY, po_id INTEGER, product_key INTEGER,
        quantity_ordered INTEGER, quantity_received INTEGER,
        unit_cost REAL, line_total REAL
    )""")
    plid = 1
    for po in range(1, 5001):
        for _ in range(rng.randint(1, 5)):
            qo = rng.randint(10, 1000)
            uc = round(rng.uniform(5, 500), 2)
            c.execute("INSERT INTO sc_po_lines VALUES (?,?,?,?,?,?,?)",
                      (plid, po, rng.randint(1, 500), qo,
                       rng.randint(0, qo), uc, round(qo * uc, 2)))
            plid += 1

    c.execute("""CREATE TABLE sc_warehouses (
        warehouse_id INTEGER PRIMARY KEY, warehouse_name TEXT, location TEXT,
        capacity_sqft INTEGER, utilization_pct REAL,
        manager_emp_key INTEGER, operating_cost REAL
    )""")
    for i, wh in enumerate(WAREHOUSES):
        c.execute("INSERT INTO sc_warehouses VALUES (?,?,?,?,?,?,?)",
                  (i + 1, wh, wh.replace("-", " "),
                   rng.randint(10000, 500000), round(rng.uniform(30, 95), 1),
                   rng.randint(1, 500), round(rng.uniform(50000, 2000000), 2)))

    c.execute("""CREATE TABLE sc_shipments (
        shipment_id INTEGER PRIMARY KEY, order_id TEXT, carrier_id INTEGER,
        ship_date TEXT, estimated_arrival TEXT, actual_arrival TEXT,
        tracking_number TEXT, status TEXT, cost REAL, weight_kg REAL
    )""")
    for i in range(15000):
        sd = _rng_date(rng)
        c.execute("INSERT INTO sc_shipments VALUES (?,?,?,?,?,?,?,?,?,?)",
                  (i + 1, f"ORD-{rng.randint(1, 50000):07d}", rng.randint(1, len(CARRIERS)),
                   sd, (datetime.strptime(sd, "%Y-%m-%d") + timedelta(days=rng.randint(1, 14))).strftime("%Y-%m-%d"),
                   (datetime.strptime(sd, "%Y-%m-%d") + timedelta(days=rng.randint(1, 21))).strftime("%Y-%m-%d") if rng.random() > 0.1 else None,
                   f"TRK{rng.randint(10 ** 11, 10 ** 12 - 1)}",
                   rng.choice(["in_transit", "delivered", "delayed", "lost", "returned"]),
                   round(rng.uniform(5, 500), 2), round(rng.uniform(0.5, 100), 2)))

    c.execute("""CREATE TABLE sc_carriers (
        carrier_id INTEGER PRIMARY KEY, carrier_name TEXT, carrier_type TEXT,
        contract_rate REAL, on_time_pct REAL, damage_rate REAL
    )""")
    for i, cr in enumerate(CARRIERS):
        c.execute("INSERT INTO sc_carriers VALUES (?,?,?,?,?,?)",
                  (i + 1, cr, rng.choice(["ground", "air", "ocean", "rail"]),
                   round(rng.uniform(2, 50), 2), round(rng.uniform(80, 99), 1),
                   round(rng.uniform(0.1, 5), 2)))

    c.execute("""CREATE TABLE sc_inventory_movements (
        movement_id INTEGER PRIMARY KEY, product_key INTEGER, warehouse_id INTEGER,
        movement_type TEXT, quantity INTEGER, movement_date TEXT, reference_id TEXT
    )""")
    for i in range(20000):
        c.execute("INSERT INTO sc_inventory_movements VALUES (?,?,?,?,?,?,?)",
                  (i + 1, rng.randint(1, 500), rng.randint(1, len(WAREHOUSES)),
                   rng.choice(["receipt", "shipment", "transfer_in", "transfer_out", "adjustment", "write_off"]),
                   rng.randint(1, 500), _rng_date(rng),
                   f"REF-{rng.randint(1, 50000):06d}"))

    c.execute("""CREATE TABLE sc_quality_inspections (
        inspection_id INTEGER PRIMARY KEY, po_id INTEGER, product_key INTEGER,
        inspection_date TEXT, pass_fail TEXT,
        defect_count INTEGER, inspector_emp_key INTEGER, notes TEXT
    )""")
    for i in range(3000):
        c.execute("INSERT INTO sc_quality_inspections VALUES (?,?,?,?,?,?,?,?)",
                  (i + 1, rng.randint(1, 5000), rng.randint(1, 500),
                   _rng_date(rng), rng.choice(["pass", "fail", "conditional"]),
                   rng.randint(0, 50), rng.randint(1, 500),
                   f"Inspection notes for batch {i + 1}"))


# ---------------------------------------------------------------------------
# 6. Customer Support Domain (cs_)  — ~8 tables
# ---------------------------------------------------------------------------

def _create_support_tables(conn, rng):
    c = conn.cursor()

    c.execute("""CREATE TABLE cs_agents (
        agent_key INTEGER PRIMARY KEY, agent_id TEXT, agent_name TEXT,
        team TEXT, skill_level INTEGER, hire_date TEXT,
        tickets_handled INTEGER, avg_satisfaction REAL
    )""")
    teams = ["Tier 1", "Tier 2", "Tier 3", "Escalation", "Billing", "Technical", "Enterprise"]
    for i in range(100):
        c.execute("INSERT INTO cs_agents VALUES (?,?,?,?,?,?,?,?)",
                  (i + 1, f"AGT-{i + 1:04d}",
                   f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}",
                   rng.choice(teams), rng.randint(1, 5), _rng_date(rng, datetime(2018, 1, 1), datetime(2024, 1, 1)),
                   rng.randint(100, 5000), round(rng.uniform(3, 5), 2)))

    c.execute("""CREATE TABLE cs_tickets (
        ticket_id INTEGER PRIMARY KEY, customer_key INTEGER, agent_key INTEGER,
        created_at TEXT, resolved_at TEXT,
        category TEXT, subcategory TEXT, priority TEXT, status TEXT,
        channel TEXT, first_response_time_min REAL,
        resolution_time_min REAL, satisfaction_score INTEGER
    )""")
    for i in range(15000):
        created = _rng_ts(rng)
        resolved = (datetime.strptime(created, "%Y-%m-%d %H:%M:%S") + timedelta(minutes=rng.randint(5, 10080))).strftime("%Y-%m-%d %H:%M:%S") if rng.random() > 0.15 else None
        c.execute("INSERT INTO cs_tickets VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                  (i + 1, rng.randint(1, 5000), rng.randint(1, 100),
                   created, resolved,
                   rng.choice(TICKET_CATEGORIES),
                   f"sub_{rng.randint(1, 5)}", rng.choice(PRIORITIES),
                   "resolved" if resolved else rng.choice(["open", "in_progress", "waiting_customer", "escalated"]),
                   rng.choice(["email", "chat", "phone", "web_form", "social"]),
                   round(rng.uniform(1, 480), 1),
                   round(rng.uniform(5, 10080), 1) if resolved else None,
                   rng.randint(1, 5) if resolved and rng.random() > 0.3 else None))

    c.execute("""CREATE TABLE cs_sla_definitions (
        sla_id INTEGER PRIMARY KEY, priority TEXT, category TEXT,
        target_first_response_min INTEGER, target_resolution_min INTEGER,
        business_hours_only INTEGER
    )""")
    sid = 1
    for pri in PRIORITIES:
        for cat in TICKET_CATEGORIES:
            base_fr = {"low": 480, "medium": 240, "high": 60, "critical": 15}[pri]
            base_res = {"low": 4320, "medium": 1440, "high": 480, "critical": 120}[pri]
            c.execute("INSERT INTO cs_sla_definitions VALUES (?,?,?,?,?,?)",
                      (sid, pri, cat, base_fr, base_res, 1 if pri != "critical" else 0))
            sid += 1

    c.execute("""CREATE TABLE cs_escalations (
        escalation_id INTEGER PRIMARY KEY, ticket_id INTEGER,
        escalated_to INTEGER, escalation_reason TEXT,
        escalated_at TEXT, resolved_at TEXT
    )""")
    for i in range(3000):
        c.execute("INSERT INTO cs_escalations VALUES (?,?,?,?,?,?)",
                  (i + 1, rng.randint(1, 15000), rng.randint(1, 100),
                   rng.choice(["sla_breach", "customer_request", "complexity", "vip_customer", "recurring_issue"]),
                   _rng_ts(rng), _rng_ts(rng) if rng.random() > 0.2 else None))

    c.execute("""CREATE TABLE cs_feedback (
        feedback_id INTEGER PRIMARY KEY, ticket_id INTEGER, customer_key INTEGER,
        rating INTEGER, comment TEXT, submitted_at TEXT
    )""")
    for i in range(8000):
        c.execute("INSERT INTO cs_feedback VALUES (?,?,?,?,?,?)",
                  (i + 1, rng.randint(1, 15000), rng.randint(1, 5000),
                   rng.randint(1, 5), f"Feedback comment {i + 1}.", _rng_ts(rng)))

    c.execute("""CREATE TABLE cs_knowledge_base (
        article_id INTEGER PRIMARY KEY, title TEXT, category TEXT,
        content_summary TEXT, views INTEGER,
        helpful_votes INTEGER, created_at TEXT, last_updated TEXT
    )""")
    for i in range(300):
        c.execute("INSERT INTO cs_knowledge_base VALUES (?,?,?,?,?,?,?,?)",
                  (i + 1, f"KB: {rng.choice(TICKET_CATEGORIES)} - FAQ #{i + 1}",
                   rng.choice(TICKET_CATEGORIES), f"Knowledge base article about {rng.choice(TICKET_CATEGORIES)}.",
                   rng.randint(10, 50000), rng.randint(0, 5000),
                   _rng_date(rng), _rng_date(rng)))

    c.execute("""CREATE TABLE cs_chat_sessions (
        session_id INTEGER PRIMARY KEY, ticket_id INTEGER,
        customer_key INTEGER, agent_key INTEGER,
        start_time TEXT, end_time TEXT,
        message_count INTEGER, sentiment REAL
    )""")
    for i in range(10000):
        st = _rng_ts(rng)
        c.execute("INSERT INTO cs_chat_sessions VALUES (?,?,?,?,?,?,?,?)",
                  (i + 1, rng.randint(1, 15000), rng.randint(1, 5000),
                   rng.randint(1, 100), st,
                   (datetime.strptime(st, "%Y-%m-%d %H:%M:%S") + timedelta(minutes=rng.randint(5, 120))).strftime("%Y-%m-%d %H:%M:%S"),
                   rng.randint(3, 80), round(rng.uniform(-1, 1), 3)))


# ---------------------------------------------------------------------------
# 7. Web Analytics Domain (wa_)  — ~8 tables
# ---------------------------------------------------------------------------

def _create_web_analytics_tables(conn, rng):
    c = conn.cursor()

    c.execute("""CREATE TABLE wa_sessions (
        session_id INTEGER PRIMARY KEY, visitor_id TEXT,
        customer_key INTEGER, start_time TEXT, end_time TEXT,
        landing_page TEXT, exit_page TEXT,
        device_type TEXT, browser TEXT, os TEXT,
        country TEXT, city TEXT,
        is_bounce INTEGER, page_count INTEGER, duration_seconds INTEGER
    )""")
    oses = ["Windows", "macOS", "iOS", "Android", "Linux"]
    for i in range(50000):
        st = _rng_ts(rng)
        dur = rng.randint(0, 3600)
        c.execute("INSERT INTO wa_sessions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                  (i + 1, f"VIS-{rng.randint(1, 30000):08d}",
                   rng.randint(1, 5000) if rng.random() > 0.4 else None,
                   st, (datetime.strptime(st, "%Y-%m-%d %H:%M:%S") + timedelta(seconds=dur)).strftime("%Y-%m-%d %H:%M:%S"),
                   rng.choice(PAGES), rng.choice(PAGES),
                   rng.choice(DEVICES), rng.choice(BROWSERS), rng.choice(oses),
                   rng.choice(COUNTRIES), f"City-{rng.randint(1, 200)}",
                   1 if dur < 10 else 0, rng.randint(1, 30) if dur > 10 else 1, dur))

    c.execute("""CREATE TABLE wa_pageviews (
        pageview_id INTEGER PRIMARY KEY, session_id INTEGER,
        page_url TEXT, page_title TEXT, timestamp TEXT,
        time_on_page_seconds INTEGER, scroll_depth_pct REAL
    )""")
    for i in range(100000):
        c.execute("INSERT INTO wa_pageviews VALUES (?,?,?,?,?,?,?)",
                  (i + 1, rng.randint(1, 50000),
                   rng.choice(PAGES) + f"/{rng.randint(1, 500)}",
                   f"Page Title {rng.randint(1, 500)}", _rng_ts(rng),
                   rng.randint(1, 300), round(rng.uniform(0, 100), 1)))

    c.execute("""CREATE TABLE wa_events (
        event_id INTEGER PRIMARY KEY, session_id INTEGER,
        event_name TEXT, event_category TEXT,
        event_label TEXT, event_value REAL, timestamp TEXT
    )""")
    evt_names = ["button_click", "form_submit", "video_play", "scroll", "download", "share", "signup_start", "signup_complete"]
    for i in range(60000):
        c.execute("INSERT INTO wa_events VALUES (?,?,?,?,?,?,?)",
                  (i + 1, rng.randint(1, 50000), rng.choice(evt_names),
                   rng.choice(["engagement", "conversion", "navigation", "interaction"]),
                   f"label_{rng.randint(1, 100)}", round(rng.uniform(0, 100), 2), _rng_ts(rng)))

    c.execute("""CREATE TABLE wa_conversions (
        conversion_id INTEGER PRIMARY KEY, session_id INTEGER,
        customer_key INTEGER, conversion_type TEXT,
        value REAL, timestamp TEXT, attribution_channel TEXT
    )""")
    for i in range(5000):
        c.execute("INSERT INTO wa_conversions VALUES (?,?,?,?,?,?,?)",
                  (i + 1, rng.randint(1, 50000), rng.randint(1, 5000),
                   rng.choice(["purchase", "signup", "demo_request", "trial_start", "download"]),
                   round(rng.uniform(10, 5000), 2), _rng_ts(rng),
                   rng.choice(CHANNELS)))

    c.execute("""CREATE TABLE wa_experiments (
        experiment_id INTEGER PRIMARY KEY, experiment_name TEXT,
        variant TEXT, metric TEXT, value REAL,
        visitors INTEGER, conversions INTEGER,
        start_date TEXT, end_date TEXT, is_winner INTEGER
    )""")
    for i in range(80):
        c.execute("INSERT INTO wa_experiments VALUES (?,?,?,?,?,?,?,?,?,?)",
                  (i + 1, f"Experiment {i + 1}", rng.choice(["control", "variant_a", "variant_b"]),
                   rng.choice(["conversion_rate", "revenue_per_visitor", "bounce_rate", "time_on_site"]),
                   round(rng.uniform(0, 0.5), 4), rng.randint(1000, 100000),
                   rng.randint(10, 5000), _rng_date(rng), _rng_date(rng), rng.randint(0, 1)))

    c.execute("""CREATE TABLE wa_search_queries (
        query_id INTEGER PRIMARY KEY, session_id INTEGER,
        search_term TEXT, results_count INTEGER,
        clicked_position INTEGER, timestamp TEXT
    )""")
    terms = ["laptop", "phone case", "returns policy", "shipping", "discount", "warranty",
             "size chart", "install guide", "api docs", "pricing", "enterprise plan"]
    for i in range(10000):
        c.execute("INSERT INTO wa_search_queries VALUES (?,?,?,?,?,?)",
                  (i + 1, rng.randint(1, 50000), rng.choice(terms) + f" {rng.choice(['', 'best', 'cheap', 'review', '2024'])}".strip(),
                   rng.randint(0, 500), rng.randint(1, 20) if rng.random() > 0.3 else None,
                   _rng_ts(rng)))

    c.execute("""CREATE TABLE wa_funnel_steps (
        step_id INTEGER PRIMARY KEY, funnel_name TEXT,
        step_number INTEGER, step_name TEXT,
        session_count INTEGER, drop_off_rate REAL
    )""")
    funnels = {
        "purchase": ["landing", "product_view", "add_to_cart", "checkout_start", "payment", "confirmation"],
        "signup": ["landing", "pricing_view", "signup_form", "email_verify", "onboarding", "activation"],
        "demo": ["landing", "features_view", "demo_form", "demo_scheduled", "demo_attended", "follow_up"],
    }
    fid = 1
    for fname, steps in funnels.items():
        count = rng.randint(5000, 50000)
        for sn, sname in enumerate(steps):
            drop = round(rng.uniform(0.1, 0.5), 3)
            c.execute("INSERT INTO wa_funnel_steps VALUES (?,?,?,?,?,?)",
                      (fid, fname, sn + 1, sname, count, drop))
            count = int(count * (1 - drop))
            fid += 1


# ---------------------------------------------------------------------------
# 8. Data Engineering Domain (de_)  — ~5 tables
# ---------------------------------------------------------------------------

def _create_data_engineering_tables(conn, rng):
    c = conn.cursor()

    c.execute("""CREATE TABLE de_etl_jobs (
        job_id INTEGER PRIMARY KEY, job_name TEXT, source_system TEXT,
        target_table TEXT, schedule TEXT,
        last_run_start TEXT, last_run_end TEXT, status TEXT,
        rows_processed INTEGER, rows_failed INTEGER
    )""")
    sources = ["salesforce", "hubspot", "stripe", "snowflake", "s3", "kafka", "postgres", "mysql", "api", "sftp"]
    for i in range(100):
        st = _rng_ts(rng)
        c.execute("INSERT INTO de_etl_jobs VALUES (?,?,?,?,?,?,?,?,?,?)",
                  (i + 1, f"etl_{rng.choice(['load', 'transform', 'sync', 'extract', 'merge'])}_{rng.choice(sources)}_{i}",
                   rng.choice(sources), f"stg_{rng.choice(['customers', 'orders', 'products', 'events', 'payments'])}",
                   rng.choice(["hourly", "daily", "weekly", "every_6h", "every_15min"]),
                   st, (datetime.strptime(st, "%Y-%m-%d %H:%M:%S") + timedelta(minutes=rng.randint(1, 120))).strftime("%Y-%m-%d %H:%M:%S"),
                   rng.choice(["success", "failed", "running", "warning"]),
                   rng.randint(0, 1000000), rng.randint(0, 1000)))

    c.execute("""CREATE TABLE de_data_quality_rules (
        rule_id INTEGER PRIMARY KEY, table_name TEXT, column_name TEXT,
        rule_type TEXT, rule_definition TEXT,
        severity TEXT, is_active INTEGER
    )""")
    rule_types = ["not_null", "unique", "range_check", "referential_integrity", "format", "freshness", "custom"]
    for i in range(200):
        c.execute("INSERT INTO de_data_quality_rules VALUES (?,?,?,?,?,?,?)",
                  (i + 1, f"{'dim' if rng.random() > 0.5 else 'fact'}_{rng.choice(['customer', 'product', 'sales', 'orders'])}",
                   f"col_{rng.randint(1, 20)}", rng.choice(rule_types),
                   f"Rule definition for {rng.choice(rule_types)}", rng.choice(["info", "warning", "error", "critical"]),
                   1 if rng.random() > 0.1 else 0))

    c.execute("""CREATE TABLE de_data_quality_results (
        result_id INTEGER PRIMARY KEY, rule_id INTEGER, run_date TEXT,
        passed INTEGER, failed_count INTEGER, total_count INTEGER, details TEXT
    )""")
    for i in range(2000):
        total = rng.randint(100, 100000)
        failed = rng.randint(0, total // 10)
        c.execute("INSERT INTO de_data_quality_results VALUES (?,?,?,?,?,?,?)",
                  (i + 1, rng.randint(1, 200), _rng_date(rng),
                   1 if failed == 0 else 0, failed, total,
                   f"DQ check details for rule {rng.randint(1, 200)}"))

    c.execute("""CREATE TABLE de_lineage (
        lineage_id INTEGER PRIMARY KEY, source_table TEXT, target_table TEXT,
        transformation TEXT, created_by TEXT
    )""")
    for i in range(300):
        c.execute("INSERT INTO de_lineage VALUES (?,?,?,?,?)",
                  (i + 1, f"raw_{rng.choice(['customers', 'orders', 'products', 'events', 'payments', 'leads'])}",
                   f"dim_{rng.choice(['customer', 'product', 'date'])}",
                   rng.choice(["join", "aggregate", "filter", "deduplicate", "pivot", "unpivot", "scd2"]),
                   f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"))

    c.execute("""CREATE TABLE de_schema_registry (
        schema_id INTEGER PRIMARY KEY, table_name TEXT, version INTEGER,
        columns_json TEXT, created_at TEXT, deprecated_at TEXT
    )""")
    for i in range(150):
        c.execute("INSERT INTO de_schema_registry VALUES (?,?,?,?,?,?)",
                  (i + 1, f"{rng.choice(['dim', 'fact', 'stg', 'raw'])}_{rng.choice(['customer', 'product', 'sales', 'orders'])}",
                   rng.randint(1, 10), '{"columns": ["col1", "col2"]}',
                   _rng_date(rng), _rng_date(rng) if rng.random() < 0.3 else None))


# ---------------------------------------------------------------------------
# 9. Reference / Lookup Tables  — ~30 tables
# ---------------------------------------------------------------------------

def _create_reference_tables(conn, rng):
    c = conn.cursor()

    ref_tables = [
        ("ref_countries", "country_code TEXT PRIMARY KEY, country_name TEXT, region TEXT, currency_code TEXT, population INTEGER"),
        ("ref_currencies", "currency_code TEXT PRIMARY KEY, currency_name TEXT, symbol TEXT, decimal_places INTEGER"),
        ("ref_states", "state_code TEXT PRIMARY KEY, state_name TEXT, country_code TEXT, timezone TEXT"),
        ("ref_product_categories", "category_id INTEGER PRIMARY KEY, category_name TEXT, parent_category_id INTEGER, is_active INTEGER"),
        ("ref_product_subcategories", "subcategory_id INTEGER PRIMARY KEY, subcategory_name TEXT, category_id INTEGER"),
        ("ref_order_statuses", "status_code TEXT PRIMARY KEY, status_name TEXT, description TEXT, is_terminal INTEGER"),
        ("ref_payment_methods", "method_code TEXT PRIMARY KEY, method_name TEXT, is_online INTEGER, processing_fee_pct REAL"),
        ("ref_shipping_methods", "method_code TEXT PRIMARY KEY, method_name TEXT, avg_days INTEGER, base_cost REAL"),
        ("ref_return_reasons", "reason_code TEXT PRIMARY KEY, reason_name TEXT, requires_inspection INTEGER"),
        ("ref_ticket_categories", "category_code TEXT PRIMARY KEY, category_name TEXT, default_priority TEXT"),
        ("ref_ticket_priorities", "priority_code TEXT PRIMARY KEY, priority_name TEXT, sla_hours INTEGER"),
        ("ref_channels", "channel_code TEXT PRIMARY KEY, channel_name TEXT, channel_type TEXT"),
        ("ref_industries", "industry_code TEXT PRIMARY KEY, industry_name TEXT, sector TEXT"),
        ("ref_job_levels", "level_id INTEGER PRIMARY KEY, level_name TEXT, min_band REAL, max_band REAL"),
        ("ref_leave_types", "type_code TEXT PRIMARY KEY, type_name TEXT, max_days INTEGER, is_paid INTEGER"),
        ("ref_warehouse_types", "type_code TEXT PRIMARY KEY, type_name TEXT, climate_controlled INTEGER"),
        ("ref_carrier_types", "type_code TEXT PRIMARY KEY, type_name TEXT, is_express INTEGER"),
        ("ref_campaign_types", "type_code TEXT PRIMARY KEY, type_name TEXT, typical_budget REAL"),
        ("ref_lead_sources", "source_code TEXT PRIMARY KEY, source_name TEXT, is_inbound INTEGER"),
        ("ref_ab_test_metrics", "metric_code TEXT PRIMARY KEY, metric_name TEXT, direction TEXT"),
        ("ref_event_types", "type_code TEXT PRIMARY KEY, type_name TEXT, is_conversion INTEGER"),
        ("ref_device_types", "type_code TEXT PRIMARY KEY, type_name TEXT"),
        ("ref_browsers", "browser_code TEXT PRIMARY KEY, browser_name TEXT, vendor TEXT"),
        ("ref_fiscal_periods", "period_id INTEGER PRIMARY KEY, fiscal_year INTEGER, fiscal_quarter INTEGER, start_date TEXT, end_date TEXT"),
        ("ref_account_types", "type_code TEXT PRIMARY KEY, type_name TEXT, normal_balance TEXT"),
        ("ref_tax_jurisdictions", "jurisdiction_id INTEGER PRIMARY KEY, name TEXT, country TEXT, tax_rate REAL"),
        ("ref_benefit_types", "type_code TEXT PRIMARY KEY, type_name TEXT, employer_match_pct REAL"),
        ("ref_seniority_levels", "level_id INTEGER PRIMARY KEY, level_name TEXT, min_years_exp INTEGER"),
        ("ref_review_ratings", "rating_id INTEGER PRIMARY KEY, rating_label TEXT, description TEXT"),
        ("ref_data_sources", "source_id INTEGER PRIMARY KEY, source_name TEXT, source_type TEXT, refresh_frequency TEXT"),
    ]

    for tname, cols in ref_tables:
        c.execute(f"CREATE TABLE {tname} ({cols})")
        # Insert a few reference rows
        col_names = [col.split()[0] for col in cols.split(",")]
        pk_col = col_names[0]
        pk_type = cols.split(",")[0].split()[1]
        for j in range(rng.randint(5, 25)):
            vals = []
            for ci, cn in enumerate(col_names):
                ct = cols.split(",")[ci].strip().split()[1] if ci < len(cols.split(",")) else "TEXT"
                if ci == 0:
                    vals.append(j + 1 if "INTEGER" in ct else f"{cn}_{j + 1}")
                elif "INTEGER" in ct:
                    vals.append(rng.randint(0, 100))
                elif "REAL" in ct:
                    vals.append(round(rng.uniform(0, 100), 2))
                else:
                    vals.append(f"{cn}_val_{j + 1}")
            placeholders = ",".join(["?"] * len(vals))
            try:
                c.execute(f"INSERT INTO {tname} VALUES ({placeholders})", vals)
            except sqlite3.IntegrityError:
                pass


# ---------------------------------------------------------------------------
# 10. Derivative Tables (stg_*, raw_*, *_history, *_archive, audit_*)
# ---------------------------------------------------------------------------

def _create_derivative_tables(conn, rng):
    """Programmatically create staging, raw, archive, history, and audit variants
    of existing tables to simulate enterprise data warehouse clutter."""
    c = conn.cursor()

    core_tables = c.execute(
        "SELECT name, sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'ref_%' AND name NOT LIKE 'stg_%' AND name NOT LIKE 'raw_%' AND name NOT LIKE 'audit_%'"
    ).fetchall()

    created = 0
    for tname, create_sql in core_tables:
        if not create_sql:
            continue

        # stg_ variant (same schema)
        stg_sql = create_sql.replace(f"CREATE TABLE {tname}", f"CREATE TABLE stg_{tname}", 1)
        try:
            c.execute(stg_sql)
            created += 1
        except sqlite3.OperationalError:
            pass

        # raw_ variant (same schema + ingestion metadata)
        raw_cols = create_sql.rstrip(")")
        raw_cols += ", _ingested_at TEXT DEFAULT CURRENT_TIMESTAMP, _source_file TEXT, _batch_id TEXT)"
        raw_sql = raw_cols.replace(f"CREATE TABLE {tname}", f"CREATE TABLE raw_{tname}", 1)
        try:
            c.execute(raw_sql)
            created += 1
        except sqlite3.OperationalError:
            pass

        # _history variant for dimension and HR tables
        if tname.startswith("dim_") or tname.startswith("hr_") or tname.startswith("sc_") or tname.startswith("cs_"):
            hist_cols = create_sql.rstrip(")")
            hist_cols += ", _valid_from TEXT, _valid_to TEXT, _is_current INTEGER DEFAULT 1)"
            hist_sql = hist_cols.replace(f"CREATE TABLE {tname}", f"CREATE TABLE {tname}_history", 1)
            # Remove PRIMARY KEY constraint to allow historical duplicates
            hist_sql = hist_sql.replace("PRIMARY KEY", "")
            try:
                c.execute(hist_sql)
                created += 1
            except sqlite3.OperationalError:
                pass

        # _archive variant for fact tables
        if tname.startswith("fact_") or tname.startswith("fin_") or tname.startswith("cs_"):
            archive_sql = create_sql.replace(f"CREATE TABLE {tname}", f"CREATE TABLE {tname}_archive", 1)
            try:
                c.execute(archive_sql)
                created += 1
            except sqlite3.OperationalError:
                pass

        # audit_ variant (subset of columns + audit metadata)
        if rng.random() > 0.2:
            audit_sql = f"""CREATE TABLE audit_{tname} (
                audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
                original_id INTEGER, action TEXT, changed_by INTEGER,
                changed_at TEXT DEFAULT CURRENT_TIMESTAMP, old_values TEXT, new_values TEXT
            )"""
            try:
                c.execute(audit_sql)
                created += 1
            except sqlite3.OperationalError:
                pass

    # Additional messy tables that real warehouses accumulate
    extras = [
        # Temp / migration artifacts
        "tmp_customer_dedup", "tmp_order_reconciliation", "tmp_revenue_fix_20231015",
        "tmp_migration_backup", "tmp_data_patch_q3", "tmp_duplicate_check_orders",
        "tmp_price_update_batch", "tmp_customer_merge_staging", "tmp_historical_load",
        "tmp_q4_reconciliation", "tmp_address_cleanup", "tmp_email_validation",
        "tmp_sku_mapping_v2", "tmp_currency_conversion_fix",
        # Backups
        "bak_dim_customer_20230601", "bak_fact_sales_pre_migration",
        "bak_hr_employees_20231201", "bak_fin_invoices_q3_fix",
        "bak_mktg_campaigns_old_schema", "bak_cs_tickets_before_recat",
        "bak_wa_sessions_deduplicated", "bak_dim_product_v1",
        # Views
        "v_monthly_revenue", "v_customer_360", "v_product_catalog",
        "v_active_inventory", "v_open_tickets", "v_employee_directory",
        "v_pipeline_summary", "v_marketing_roi", "v_supplier_scorecard",
        "v_daily_kpi_dashboard", "v_customer_health_score",
        "v_product_margin_analysis", "v_sales_by_region",
        "v_support_sla_compliance", "v_funnel_conversion",
        # Reports
        "rpt_daily_sales", "rpt_weekly_support", "rpt_monthly_financials",
        "rpt_quarterly_hr", "rpt_annual_review", "rpt_customer_churn_monthly",
        "rpt_product_performance_weekly", "rpt_marketing_spend_daily",
        "rpt_inventory_levels_daily", "rpt_revenue_by_geo_quarterly",
        "rpt_employee_headcount_monthly", "rpt_support_volume_hourly",
        "rpt_web_traffic_daily", "rpt_supplier_delivery_weekly",
        # Logs
        "log_api_calls", "log_user_logins", "log_data_exports",
        "log_schema_changes", "log_permission_changes", "log_query_history",
        "log_etl_errors", "log_alert_notifications", "log_model_predictions",
        "log_feature_flags", "log_ab_test_assignments", "log_cache_invalidations",
        # Caches
        "cache_customer_scores", "cache_product_recommendations",
        "cache_search_results", "cache_dashboard_data",
        "cache_segment_memberships", "cache_pricing_rules",
        "cache_inventory_availability", "cache_geo_lookups",
        # External system syncs
        "ext_salesforce_contacts", "ext_salesforce_opportunities",
        "ext_salesforce_accounts", "ext_hubspot_deals", "ext_hubspot_contacts",
        "ext_stripe_charges", "ext_stripe_subscriptions",
        "ext_google_analytics", "ext_google_ads_campaigns",
        "ext_zendesk_tickets", "ext_zendesk_users",
        "ext_jira_issues", "ext_slack_messages",
        "ext_snowflake_usage", "ext_aws_cost_explorer",
        # Sandbox / analyst
        "sandbox_analyst_query_1", "sandbox_analyst_query_2",
        "sandbox_ml_features", "sandbox_test_data",
        "sandbox_churn_model_v3", "sandbox_ltv_prediction",
        "sandbox_segment_experiment", "sandbox_price_elasticity",
        # Deprecated
        "deprecated_old_customers", "deprecated_legacy_orders",
        "deprecated_v1_products", "deprecated_manual_reports",
        "deprecated_old_ticket_system", "deprecated_legacy_inventory",
        "deprecated_v2_attribution", "deprecated_old_payroll",
        # Snapshots / materialized
        "snap_customer_daily", "snap_inventory_weekly",
        "snap_sales_monthly", "snap_web_metrics_daily",
        "snap_support_sla_daily", "snap_marketing_spend_weekly",
        "mat_customer_lifetime_stats", "mat_product_affinity_matrix",
        "mat_channel_attribution_summary", "mat_cohort_retention_monthly",
        "mat_department_budget_vs_actual", "mat_supplier_performance_quarterly",
        # Data science / ML pipeline tables
        "ml_feature_store_customer", "ml_feature_store_product",
        "ml_feature_store_session", "ml_training_runs",
        "ml_model_registry", "ml_predictions_churn",
        "ml_predictions_ltv", "ml_predictions_next_purchase",
        "ml_experiment_results", "ml_embeddings_product",
        # Compliance / regulatory
        "compliance_gdpr_requests", "compliance_data_retention_log",
        "compliance_access_audit", "compliance_pii_inventory",
        "compliance_consent_records", "compliance_incident_reports",
        # Partner / B2B integration
        "partner_api_keys", "partner_webhook_events",
        "partner_revenue_share", "partner_onboarding_status",
        "partner_tier_history", "partner_commission_payouts",
        # Pricing / billing engine
        "billing_subscriptions", "billing_usage_records",
        "billing_credits", "billing_invoice_drafts",
        "pricing_rules_active", "pricing_overrides",
        "pricing_tier_definitions", "pricing_ab_test_prices",
    ]
    for tname in extras:
        try:
            if tname.startswith("v_"):
                c.execute(f"CREATE VIEW {tname} AS SELECT 1 AS placeholder")
            else:
                c.execute(f"""CREATE TABLE {tname} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    data TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    notes TEXT
                )""")
            created += 1
        except sqlite3.OperationalError:
            pass

    conn.commit()
    return created


# ---------------------------------------------------------------------------
# 11. Create indexes for realistic query performance
# ---------------------------------------------------------------------------

def _create_indexes(conn):
    indexes = [
        "CREATE INDEX IF NOT EXISTS idx_fs_customer ON fact_sales(customer_key)",
        "CREATE INDEX IF NOT EXISTS idx_fs_product ON fact_sales(product_key)",
        "CREATE INDEX IF NOT EXISTS idx_fs_date ON fact_sales(date_key)",
        "CREATE INDEX IF NOT EXISTS idx_fs_order_date ON fact_sales(order_date)",
        "CREATE INDEX IF NOT EXISTS idx_fs_status ON fact_sales(status)",
        "CREATE INDEX IF NOT EXISTS idx_fr_sale ON fact_returns(sale_id)",
        "CREATE INDEX IF NOT EXISTS idx_fr_customer ON fact_returns(customer_key)",
        "CREATE INDEX IF NOT EXISTS idx_fwt_customer ON fact_web_traffic(customer_key)",
        "CREATE INDEX IF NOT EXISTS idx_fwt_session ON fact_web_traffic(session_id)",
        "CREATE INDEX IF NOT EXISTS idx_fi_invoice_cust ON fin_invoices(customer_key)",
        "CREATE INDEX IF NOT EXISTS idx_fi_status ON fin_invoices(status)",
        "CREATE INDEX IF NOT EXISTS idx_fil_invoice ON fin_invoice_lines(invoice_id)",
        "CREATE INDEX IF NOT EXISTS idx_fp_invoice ON fin_payments(invoice_id)",
        "CREATE INDEX IF NOT EXISTS idx_he_dept ON hr_employees(department_key)",
        "CREATE INDEX IF NOT EXISTS idx_he_manager ON hr_employees(manager_key)",
        "CREATE INDEX IF NOT EXISTS idx_hp_emp ON hr_payroll(emp_key)",
        "CREATE INDEX IF NOT EXISTS idx_ml_campaign ON mktg_leads(campaign_id)",
        "CREATE INDEX IF NOT EXISTS idx_ma_customer ON mktg_attribution(customer_key)",
        "CREATE INDEX IF NOT EXISTS idx_ma_campaign ON mktg_attribution(campaign_id)",
        "CREATE INDEX IF NOT EXISTS idx_spo_supplier ON sc_purchase_orders(supplier_key)",
        "CREATE INDEX IF NOT EXISTS idx_ss_order ON sc_shipments(order_id)",
        "CREATE INDEX IF NOT EXISTS idx_ct_customer ON cs_tickets(customer_key)",
        "CREATE INDEX IF NOT EXISTS idx_ct_agent ON cs_tickets(agent_key)",
        "CREATE INDEX IF NOT EXISTS idx_ct_created ON cs_tickets(created_at)",
        "CREATE INDEX IF NOT EXISTS idx_ws_customer ON wa_sessions(customer_key)",
        "CREATE INDEX IF NOT EXISTS idx_wp_session ON wa_pageviews(session_id)",
        "CREATE INDEX IF NOT EXISTS idx_we_session ON wa_events(session_id)",
    ]
    for idx in indexes:
        try:
            conn.execute(idx)
        except sqlite3.OperationalError:
            pass
    conn.commit()


# ---------------------------------------------------------------------------
# 12. Main DB creation orchestrator
# ---------------------------------------------------------------------------

def create_warehouse_db(db_path: str, seed: int = 42):
    rng = random.Random(seed)
    if os.path.exists(db_path):
        os.remove(db_path)

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")

    print("Creating core business tables...")
    n_products = _create_core_tables(conn, rng)
    conn.commit()

    print("Creating finance tables...")
    _create_finance_tables(conn, rng)
    conn.commit()

    print("Creating HR tables...")
    _create_hr_tables(conn, rng)
    conn.commit()

    print("Creating marketing tables...")
    _create_marketing_tables(conn, rng)
    conn.commit()

    print("Creating supply chain tables...")
    _create_supply_chain_tables(conn, rng)
    conn.commit()

    print("Creating customer support tables...")
    _create_support_tables(conn, rng)
    conn.commit()

    print("Creating web analytics tables...")
    _create_web_analytics_tables(conn, rng)
    conn.commit()

    print("Creating data engineering tables...")
    _create_data_engineering_tables(conn, rng)
    conn.commit()

    print("Creating reference tables...")
    _create_reference_tables(conn, rng)
    conn.commit()

    print("Creating derivative tables (stg_, raw_, _history, _archive, audit_)...")
    n_derived = _create_derivative_tables(conn, rng)
    conn.commit()

    print("Creating indexes...")
    _create_indexes(conn)

    # Print summary
    tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    views = conn.execute("SELECT name FROM sqlite_master WHERE type='view'").fetchall()
    print(f"\nWarehouse created at {db_path}")
    print(f"  Tables: {len(tables)}")
    print(f"  Views:  {len(views)}")
    print(f"  Total:  {len(tables) + len(views)}")

    for t, in tables[:10]:
        try:
            cnt = conn.execute(f"SELECT COUNT(*) FROM \"{t}\"").fetchone()[0]
            print(f"  {t}: {cnt} rows")
        except Exception:
            pass
    print("  ...")

    conn.close()
    return len(tables) + len(views)


# ---------------------------------------------------------------------------
# 13. Complex NL2SQL Prompts with Gold SQL
# ---------------------------------------------------------------------------

GOLD_QUERIES = [
    {
        "task_id": "warehouse_001",
        "difficulty": "hard",
        "question": "What's the quarter-over-quarter revenue change by product line for customers who churned in the last 90 days? A churned customer is one whose last order was more than 90 days before 2024-12-31.",
        "gold_sql": """
WITH churned_customers AS (
    SELECT fs.customer_key
    FROM fact_sales fs
    WHERE fs.status = 'completed'
    GROUP BY fs.customer_key
    HAVING MAX(fs.order_date) < date('2024-12-31', '-90 days')
),
quarterly_rev AS (
    SELECT
        dp.product_line,
        CAST(strftime('%Y', fs.order_date) AS INTEGER) AS yr,
        ((CAST(strftime('%m', fs.order_date) AS INTEGER) - 1) / 3 + 1) AS qtr,
        SUM(fs.revenue) AS revenue
    FROM fact_sales fs
    JOIN dim_product dp ON fs.product_key = dp.product_key
    WHERE fs.customer_key IN (SELECT customer_key FROM churned_customers)
      AND fs.status = 'completed'
    GROUP BY dp.product_line, yr, qtr
)
SELECT
    product_line, yr, qtr, revenue,
    LAG(revenue) OVER (PARTITION BY product_line ORDER BY yr, qtr) AS prev_qtr_revenue,
    ROUND((revenue - LAG(revenue) OVER (PARTITION BY product_line ORDER BY yr, qtr))
          * 100.0 / LAG(revenue) OVER (PARTITION BY product_line ORDER BY yr, qtr), 2) AS qoq_change_pct
FROM quarterly_rev
ORDER BY product_line, yr, qtr
""",
        "tables_needed": ["fact_sales", "dim_product"],
    },
    {
        "task_id": "warehouse_002",
        "difficulty": "hard",
        "question": "Calculate customer acquisition cost (CAC) by marketing channel. CAC = total campaign spend for a channel / number of customers whose first purchase was attributed to that channel. Only consider customers who actually made at least one purchase.",
        "gold_sql": """
WITH first_purchase AS (
    SELECT
        fs.customer_key,
        MIN(fs.order_date) AS first_order_date
    FROM fact_sales fs
    WHERE fs.status = 'completed'
    GROUP BY fs.customer_key
),
customer_channel AS (
    SELECT
        fp.customer_key,
        dc.channel_name
    FROM first_purchase fp
    JOIN fact_sales fs ON fp.customer_key = fs.customer_key
        AND fs.order_date = fp.first_order_date
        AND fs.status = 'completed'
    JOIN dim_channel dc ON fs.channel_key = dc.channel_key
    GROUP BY fp.customer_key, dc.channel_name
),
channel_spend AS (
    SELECT
        dc.channel_name,
        SUM(mc.spend) AS total_spend
    FROM mktg_campaigns mc
    JOIN dim_channel dc ON mc.channel_key = dc.channel_key
    GROUP BY dc.channel_name
),
channel_customers AS (
    SELECT channel_name, COUNT(DISTINCT customer_key) AS acquired_customers
    FROM customer_channel
    GROUP BY channel_name
)
SELECT
    cs.channel_name,
    cs.total_spend,
    COALESCE(cc.acquired_customers, 0) AS acquired_customers,
    CASE WHEN COALESCE(cc.acquired_customers, 0) > 0
         THEN ROUND(cs.total_spend / cc.acquired_customers, 2)
         ELSE NULL END AS cac
FROM channel_spend cs
LEFT JOIN channel_customers cc ON cs.channel_name = cc.channel_name
ORDER BY cac
""",
        "tables_needed": ["fact_sales", "dim_channel", "mktg_campaigns"],
    },
    {
        "task_id": "warehouse_003",
        "difficulty": "hard",
        "question": "Which product categories have declining support satisfaction scores but increasing sales revenue? Compare Q3 2024 vs Q4 2024 for both metrics.",
        "gold_sql": """
WITH q3_sales AS (
    SELECT dp.category, SUM(fs.revenue) AS revenue
    FROM fact_sales fs
    JOIN dim_product dp ON fs.product_key = dp.product_key
    JOIN dim_date dd ON fs.date_key = dd.date_key
    WHERE dd.year = 2024 AND dd.quarter = 3 AND fs.status = 'completed'
    GROUP BY dp.category
),
q4_sales AS (
    SELECT dp.category, SUM(fs.revenue) AS revenue
    FROM fact_sales fs
    JOIN dim_product dp ON fs.product_key = dp.product_key
    JOIN dim_date dd ON fs.date_key = dd.date_key
    WHERE dd.year = 2024 AND dd.quarter = 4 AND fs.status = 'completed'
    GROUP BY dp.category
),
q3_satisfaction AS (
    SELECT dp.category, AVG(ct.satisfaction_score) AS avg_score
    FROM cs_tickets ct
    JOIN fact_sales fs ON ct.customer_key = fs.customer_key
    JOIN dim_product dp ON fs.product_key = dp.product_key
    WHERE ct.created_at >= '2024-07-01' AND ct.created_at < '2024-10-01'
      AND ct.satisfaction_score IS NOT NULL
    GROUP BY dp.category
),
q4_satisfaction AS (
    SELECT dp.category, AVG(ct.satisfaction_score) AS avg_score
    FROM cs_tickets ct
    JOIN fact_sales fs ON ct.customer_key = fs.customer_key
    JOIN dim_product dp ON fs.product_key = dp.product_key
    WHERE ct.created_at >= '2024-10-01' AND ct.created_at < '2025-01-01'
      AND ct.satisfaction_score IS NOT NULL
    GROUP BY dp.category
)
SELECT
    q3s.category,
    q3s.revenue AS q3_revenue, q4s.revenue AS q4_revenue,
    ROUND((q4s.revenue - q3s.revenue) * 100.0 / q3s.revenue, 2) AS revenue_change_pct,
    q3sat.avg_score AS q3_satisfaction, q4sat.avg_score AS q4_satisfaction,
    ROUND(q4sat.avg_score - q3sat.avg_score, 2) AS satisfaction_change
FROM q3_sales q3s
JOIN q4_sales q4s ON q3s.category = q4s.category
JOIN q3_satisfaction q3sat ON q3s.category = q3sat.category
JOIN q4_satisfaction q4sat ON q3s.category = q4sat.category
WHERE q4s.revenue > q3s.revenue AND q4sat.avg_score < q3sat.avg_score
ORDER BY revenue_change_pct DESC
""",
        "tables_needed": ["fact_sales", "dim_product", "dim_date", "cs_tickets"],
    },
    {
        "task_id": "warehouse_004",
        "difficulty": "hard",
        "question": "Calculate the true cost of returns by supplier, including the original cost of goods, refund amounts, and restocking fees. Rank suppliers by net loss (refund_amount - restocking_fee + cost_of_goods for returned items).",
        "gold_sql": """
SELECT
    ss.supplier_name, ss.country, ss.rating,
    COUNT(fr.return_id) AS return_count,
    ROUND(SUM(fr.refund_amount), 2) AS total_refunds,
    ROUND(SUM(fr.restocking_fee), 2) AS total_restocking_fees,
    ROUND(SUM(fs.cost_of_goods), 2) AS total_cogs_returned,
    ROUND(SUM(fr.refund_amount) - SUM(fr.restocking_fee) + SUM(fs.cost_of_goods), 2) AS net_loss
FROM fact_returns fr
JOIN fact_sales fs ON fr.sale_id = fs.sale_id
JOIN dim_product dp ON fr.product_key = dp.product_key
JOIN sc_suppliers ss ON dp.supplier_key = ss.supplier_key
GROUP BY ss.supplier_key, ss.supplier_name, ss.country, ss.rating
ORDER BY net_loss DESC
LIMIT 20
""",
        "tables_needed": ["fact_returns", "fact_sales", "dim_product", "sc_suppliers"],
    },
    {
        "task_id": "warehouse_005",
        "difficulty": "hard",
        "question": "Identify employees whose teams' average support ticket resolution time is more than 2x the company-wide average. For these teams, also show the turnover rate (terminated/total employees).",
        "gold_sql": """
WITH company_avg AS (
    SELECT AVG(resolution_time_min) AS avg_resolution
    FROM cs_tickets
    WHERE resolution_time_min IS NOT NULL
),
team_resolution AS (
    SELECT
        ca.team,
        AVG(ct.resolution_time_min) AS team_avg_resolution,
        COUNT(DISTINCT ct.ticket_id) AS tickets_handled
    FROM cs_tickets ct
    JOIN cs_agents ca ON ct.agent_key = ca.agent_key
    WHERE ct.resolution_time_min IS NOT NULL
    GROUP BY ca.team
),
dept_turnover AS (
    SELECT
        hd.department_name,
        COUNT(*) AS total_employees,
        SUM(CASE WHEN he.employment_status = 'terminated' THEN 1 ELSE 0 END) AS terminated,
        ROUND(SUM(CASE WHEN he.employment_status = 'terminated' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS turnover_pct
    FROM hr_employees he
    JOIN hr_departments hd ON he.department_key = hd.department_key
    GROUP BY hd.department_name
)
SELECT
    tr.team,
    tr.team_avg_resolution,
    ca2.avg_resolution AS company_avg_resolution,
    ROUND(tr.team_avg_resolution / ca2.avg_resolution, 2) AS ratio_to_avg,
    tr.tickets_handled,
    dt.department_name,
    dt.total_employees,
    dt.turnover_pct
FROM team_resolution tr
CROSS JOIN company_avg ca2
LEFT JOIN dept_turnover dt ON tr.team = dt.department_name
WHERE tr.team_avg_resolution > 2 * ca2.avg_resolution
ORDER BY tr.team_avg_resolution DESC
""",
        "tables_needed": ["cs_tickets", "cs_agents", "hr_employees", "hr_departments"],
    },
    {
        "task_id": "warehouse_006",
        "difficulty": "medium",
        "question": "What's the average time (in days) between a customer's first and second purchase? Break down by customer tier and acquisition channel.",
        "gold_sql": """
WITH ranked_orders AS (
    SELECT
        fs.customer_key,
        fs.order_date,
        ROW_NUMBER() OVER (PARTITION BY fs.customer_key ORDER BY fs.order_date) AS order_rank
    FROM fact_sales fs
    WHERE fs.status = 'completed'
),
first_second AS (
    SELECT
        r1.customer_key,
        r1.order_date AS first_order,
        r2.order_date AS second_order,
        JULIANDAY(r2.order_date) - JULIANDAY(r1.order_date) AS days_between
    FROM ranked_orders r1
    JOIN ranked_orders r2 ON r1.customer_key = r2.customer_key
        AND r1.order_rank = 1 AND r2.order_rank = 2
)
SELECT
    dc.tier,
    dc.acquisition_channel,
    COUNT(*) AS customer_count,
    ROUND(AVG(fs2.days_between), 1) AS avg_days_to_second_order,
    ROUND(MIN(fs2.days_between), 1) AS min_days,
    ROUND(MAX(fs2.days_between), 1) AS max_days
FROM first_second fs2
JOIN dim_customer dc ON fs2.customer_key = dc.customer_key
GROUP BY dc.tier, dc.acquisition_channel
HAVING COUNT(*) >= 5
ORDER BY avg_days_to_second_order
""",
        "tables_needed": ["fact_sales", "dim_customer"],
    },
    {
        "task_id": "warehouse_007",
        "difficulty": "medium",
        "question": "Find the top 10 products that are frequently bought together (appear in the same order). Show product pairs with their co-occurrence count and combined revenue.",
        "gold_sql": """
WITH order_products AS (
    SELECT DISTINCT order_id, product_key
    FROM fact_sales
    WHERE status = 'completed'
)
SELECT
    dp1.name AS product_1,
    dp2.name AS product_2,
    COUNT(*) AS co_occurrence_count,
    ROUND(SUM(fs1.revenue + fs2.revenue), 2) AS combined_revenue
FROM order_products op1
JOIN order_products op2 ON op1.order_id = op2.order_id AND op1.product_key < op2.product_key
JOIN dim_product dp1 ON op1.product_key = dp1.product_key
JOIN dim_product dp2 ON op2.product_key = dp2.product_key
JOIN fact_sales fs1 ON op1.order_id = fs1.order_id AND op1.product_key = fs1.product_key AND fs1.status = 'completed'
JOIN fact_sales fs2 ON op2.order_id = fs2.order_id AND op2.product_key = fs2.product_key AND fs2.status = 'completed'
GROUP BY dp1.name, dp2.name
ORDER BY co_occurrence_count DESC
LIMIT 10
""",
        "tables_needed": ["fact_sales", "dim_product"],
    },
    {
        "task_id": "warehouse_008",
        "difficulty": "hard",
        "question": "Build a customer cohort retention analysis. For each signup quarter (2023 Q1-Q4), calculate the percentage of customers who made at least one purchase in each subsequent quarter.",
        "gold_sql": """
WITH signup_cohorts AS (
    SELECT
        customer_key,
        CAST(strftime('%Y', signup_date) AS INTEGER) AS signup_year,
        ((CAST(strftime('%m', signup_date) AS INTEGER) - 1) / 3 + 1) AS signup_quarter
    FROM dim_customer
    WHERE signup_date >= '2023-01-01' AND signup_date < '2024-01-01'
),
quarterly_activity AS (
    SELECT DISTINCT
        fs.customer_key,
        CAST(strftime('%Y', fs.order_date) AS INTEGER) AS activity_year,
        ((CAST(strftime('%m', fs.order_date) AS INTEGER) - 1) / 3 + 1) AS activity_quarter
    FROM fact_sales fs
    WHERE fs.status = 'completed'
),
cohort_sizes AS (
    SELECT signup_year, signup_quarter, COUNT(*) AS cohort_size
    FROM signup_cohorts
    GROUP BY signup_year, signup_quarter
),
retention AS (
    SELECT
        sc.signup_year, sc.signup_quarter,
        qa.activity_year, qa.activity_quarter,
        COUNT(DISTINCT sc.customer_key) AS active_customers
    FROM signup_cohorts sc
    JOIN quarterly_activity qa ON sc.customer_key = qa.customer_key
    WHERE (qa.activity_year * 4 + qa.activity_quarter) >= (sc.signup_year * 4 + sc.signup_quarter)
    GROUP BY sc.signup_year, sc.signup_quarter, qa.activity_year, qa.activity_quarter
)
SELECT
    r.signup_year, r.signup_quarter,
    r.activity_year, r.activity_quarter,
    cs.cohort_size,
    r.active_customers,
    ROUND(r.active_customers * 100.0 / cs.cohort_size, 2) AS retention_pct
FROM retention r
JOIN cohort_sizes cs ON r.signup_year = cs.signup_year AND r.signup_quarter = cs.signup_quarter
ORDER BY r.signup_year, r.signup_quarter, r.activity_year, r.activity_quarter
""",
        "tables_needed": ["dim_customer", "fact_sales"],
    },
    {
        "task_id": "warehouse_009",
        "difficulty": "hard",
        "question": "Calculate the conversion funnel from web sessions: landing -> product_view -> add_to_cart -> checkout_start -> purchase. Show conversion rate at each step, broken down by device type.",
        "gold_sql": """
WITH session_events AS (
    SELECT
        ws.session_id,
        ws.device_type,
        MAX(CASE WHEN we.event_name = 'button_click' OR ws.page_count > 0 THEN 1 ELSE 0 END) AS landed,
        MAX(CASE WHEN wp.page_url LIKE '%product%' THEN 1 ELSE 0 END) AS viewed_product,
        MAX(CASE WHEN fwt.event_type = 'add_to_cart' THEN 1 ELSE 0 END) AS added_to_cart,
        MAX(CASE WHEN fwt.event_type = 'checkout_start' THEN 1 ELSE 0 END) AS started_checkout,
        MAX(CASE WHEN fwt.event_type = 'purchase' THEN 1 ELSE 0 END) AS purchased
    FROM wa_sessions ws
    LEFT JOIN wa_events we ON ws.session_id = we.session_id
    LEFT JOIN wa_pageviews wp ON ws.session_id = wp.session_id
    LEFT JOIN fact_web_traffic fwt ON ws.session_id = CAST(fwt.session_id AS INTEGER)
    GROUP BY ws.session_id, ws.device_type
)
SELECT
    device_type,
    COUNT(*) AS total_sessions,
    SUM(viewed_product) AS product_views,
    ROUND(SUM(viewed_product) * 100.0 / COUNT(*), 2) AS view_rate,
    SUM(added_to_cart) AS cart_adds,
    ROUND(SUM(added_to_cart) * 100.0 / NULLIF(SUM(viewed_product), 0), 2) AS add_to_cart_rate,
    SUM(started_checkout) AS checkout_starts,
    ROUND(SUM(started_checkout) * 100.0 / NULLIF(SUM(added_to_cart), 0), 2) AS checkout_rate,
    SUM(purchased) AS purchases,
    ROUND(SUM(purchased) * 100.0 / NULLIF(SUM(started_checkout), 0), 2) AS purchase_rate,
    ROUND(SUM(purchased) * 100.0 / COUNT(*), 2) AS overall_conversion_rate
FROM session_events
GROUP BY device_type
ORDER BY overall_conversion_rate DESC
""",
        "tables_needed": ["wa_sessions", "wa_events", "wa_pageviews", "fact_web_traffic"],
    },
    {
        "task_id": "warehouse_010",
        "difficulty": "medium",
        "question": "Find departments where the actual salary spend exceeds the allocated budget. Show department name, total salaries, budget, and the overage percentage.",
        "gold_sql": """
SELECT
    hd.department_name,
    COUNT(he.emp_key) AS employee_count,
    ROUND(SUM(he.salary), 2) AS total_salary_spend,
    hd.budget AS department_budget,
    ROUND(SUM(he.salary) - hd.budget, 2) AS overage,
    ROUND((SUM(he.salary) - hd.budget) * 100.0 / hd.budget, 2) AS overage_pct
FROM hr_employees he
JOIN hr_departments hd ON he.department_key = hd.department_key
WHERE he.employment_status = 'active'
GROUP BY hd.department_key, hd.department_name, hd.budget
HAVING SUM(he.salary) > hd.budget
ORDER BY overage_pct DESC
""",
        "tables_needed": ["hr_employees", "hr_departments"],
    },
    {
        "task_id": "warehouse_011",
        "difficulty": "hard",
        "question": "What's the customer lifetime value (CLV) segmented by industry and acquisition channel? CLV = total revenue - total refunds. Show only segments with at least 10 customers. Include the average time to first purchase from signup.",
        "gold_sql": """
WITH customer_revenue AS (
    SELECT
        fs.customer_key,
        SUM(fs.revenue) AS total_revenue,
        MIN(fs.order_date) AS first_order_date
    FROM fact_sales fs
    WHERE fs.status = 'completed'
    GROUP BY fs.customer_key
),
customer_refunds AS (
    SELECT customer_key, SUM(refund_amount) AS total_refunds
    FROM fact_returns
    GROUP BY customer_key
)
SELECT
    dc.industry,
    dc.acquisition_channel,
    COUNT(DISTINCT dc.customer_key) AS customer_count,
    ROUND(AVG(COALESCE(cr.total_revenue, 0) - COALESCE(crf.total_refunds, 0)), 2) AS avg_clv,
    ROUND(SUM(COALESCE(cr.total_revenue, 0) - COALESCE(crf.total_refunds, 0)), 2) AS total_clv,
    ROUND(AVG(JULIANDAY(cr.first_order_date) - JULIANDAY(dc.signup_date)), 1) AS avg_days_to_first_purchase
FROM dim_customer dc
LEFT JOIN customer_revenue cr ON dc.customer_key = cr.customer_key
LEFT JOIN customer_refunds crf ON dc.customer_key = crf.customer_key
WHERE cr.total_revenue IS NOT NULL
GROUP BY dc.industry, dc.acquisition_channel
HAVING COUNT(DISTINCT dc.customer_key) >= 10
ORDER BY avg_clv DESC
""",
        "tables_needed": ["dim_customer", "fact_sales", "fact_returns"],
    },
    {
        "task_id": "warehouse_012",
        "difficulty": "medium",
        "question": "Show the monthly trend of overdue invoices (status = 'overdue') for the year 2024. Include count, total amount, and average days outstanding.",
        "gold_sql": """
SELECT
    CAST(strftime('%m', fi.due_date) AS INTEGER) AS month,
    COUNT(*) AS overdue_count,
    ROUND(SUM(fi.total_amount), 2) AS total_overdue_amount,
    ROUND(AVG(far.days_outstanding), 1) AS avg_days_outstanding
FROM fin_invoices fi
LEFT JOIN fin_accounts_receivable far ON fi.invoice_id = far.invoice_id
WHERE fi.status = 'overdue'
  AND fi.due_date >= '2024-01-01' AND fi.due_date < '2025-01-01'
GROUP BY month
ORDER BY month
""",
        "tables_needed": ["fin_invoices", "fin_accounts_receivable"],
    },
    {
        "task_id": "warehouse_013",
        "difficulty": "hard",
        "question": "Analyze the email marketing funnel: for each campaign type, what's the average open rate, click-through rate (clicks/opens), and what's the conversion rate for recipients who clicked (matched by email to customer purchases within 7 days)?",
        "gold_sql": """
WITH email_stats AS (
    SELECT
        mc.campaign_type,
        mc.campaign_id,
        COUNT(*) AS total_sends,
        SUM(mes.delivered) AS delivered,
        SUM(mes.opened) AS opened,
        SUM(mes.clicked) AS clicked
    FROM mktg_email_sends mes
    JOIN mktg_campaigns mc ON mes.campaign_id = mc.campaign_id
    GROUP BY mc.campaign_type, mc.campaign_id
)
SELECT
    campaign_type,
    SUM(total_sends) AS total_sends,
    SUM(delivered) AS total_delivered,
    ROUND(SUM(opened) * 100.0 / NULLIF(SUM(delivered), 0), 2) AS open_rate_pct,
    ROUND(SUM(clicked) * 100.0 / NULLIF(SUM(opened), 0), 2) AS click_through_rate_pct,
    COUNT(DISTINCT campaign_id) AS campaign_count
FROM email_stats
GROUP BY campaign_type
ORDER BY open_rate_pct DESC
""",
        "tables_needed": ["mktg_email_sends", "mktg_campaigns"],
    },
    {
        "task_id": "warehouse_014",
        "difficulty": "hard",
        "question": "Which warehouses have the most inventory discrepancies? Compare the current inventory snapshot (quantity_on_hand) against the net inventory movements (receipts - shipments - write_offs) to find mismatches.",
        "gold_sql": """
WITH movement_totals AS (
    SELECT
        warehouse_id,
        product_key,
        SUM(CASE WHEN movement_type IN ('receipt', 'transfer_in') THEN quantity ELSE 0 END) AS total_in,
        SUM(CASE WHEN movement_type IN ('shipment', 'transfer_out', 'write_off') THEN quantity ELSE 0 END) AS total_out
    FROM sc_inventory_movements
    GROUP BY warehouse_id, product_key
),
expected_vs_actual AS (
    SELECT
        sw.warehouse_name,
        fis.product_key,
        fis.quantity_on_hand AS actual_qty,
        COALESCE(mt.total_in, 0) - COALESCE(mt.total_out, 0) AS expected_qty,
        fis.quantity_on_hand - (COALESCE(mt.total_in, 0) - COALESCE(mt.total_out, 0)) AS discrepancy
    FROM fact_inventory_snapshot fis
    JOIN sc_warehouses sw ON fis.store_key = sw.warehouse_id
    LEFT JOIN movement_totals mt ON fis.store_key = mt.warehouse_id AND fis.product_key = mt.product_key
)
SELECT
    warehouse_name,
    COUNT(*) AS products_with_discrepancy,
    SUM(ABS(discrepancy)) AS total_abs_discrepancy,
    ROUND(AVG(discrepancy), 2) AS avg_discrepancy,
    SUM(CASE WHEN discrepancy > 0 THEN 1 ELSE 0 END) AS over_count,
    SUM(CASE WHEN discrepancy < 0 THEN 1 ELSE 0 END) AS under_count
FROM expected_vs_actual
WHERE discrepancy != 0
GROUP BY warehouse_name
ORDER BY total_abs_discrepancy DESC
""",
        "tables_needed": ["fact_inventory_snapshot", "sc_inventory_movements", "sc_warehouses"],
    },
    {
        "task_id": "warehouse_015",
        "difficulty": "hard",
        "question": "For the top 5 marketing campaigns by spend, what was the ROI? Calculate as (attributed_revenue - campaign_spend) / campaign_spend. Use last-touch attribution model.",
        "gold_sql": """
WITH top_campaigns AS (
    SELECT campaign_id, campaign_name, campaign_type, spend
    FROM mktg_campaigns
    ORDER BY spend DESC
    LIMIT 5
),
attributed_rev AS (
    SELECT
        ma.campaign_id,
        SUM(ma.revenue_attributed) AS total_attributed_revenue
    FROM mktg_attribution ma
    WHERE ma.attribution_model = 'last_touch'
      AND ma.campaign_id IN (SELECT campaign_id FROM top_campaigns)
    GROUP BY ma.campaign_id
)
SELECT
    tc.campaign_id, tc.campaign_name, tc.campaign_type,
    ROUND(tc.spend, 2) AS spend,
    ROUND(COALESCE(ar.total_attributed_revenue, 0), 2) AS attributed_revenue,
    ROUND((COALESCE(ar.total_attributed_revenue, 0) - tc.spend) / tc.spend * 100, 2) AS roi_pct
FROM top_campaigns tc
LEFT JOIN attributed_rev ar ON tc.campaign_id = ar.campaign_id
ORDER BY roi_pct DESC
""",
        "tables_needed": ["mktg_campaigns", "mktg_attribution"],
    },
    {
        "task_id": "warehouse_016",
        "difficulty": "medium",
        "question": "What's the supplier on-time delivery rate? For each supplier, compute the percentage of purchase orders where actual_delivery <= expected_delivery.",
        "gold_sql": """
SELECT
    ss.supplier_name,
    ss.country,
    ss.rating,
    COUNT(*) AS total_pos,
    SUM(CASE WHEN spo.actual_delivery IS NOT NULL AND spo.actual_delivery <= spo.expected_delivery THEN 1 ELSE 0 END) AS on_time,
    ROUND(SUM(CASE WHEN spo.actual_delivery IS NOT NULL AND spo.actual_delivery <= spo.expected_delivery THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS on_time_pct,
    ROUND(AVG(JULIANDAY(spo.actual_delivery) - JULIANDAY(spo.expected_delivery)), 1) AS avg_delay_days
FROM sc_purchase_orders spo
JOIN sc_suppliers ss ON spo.supplier_key = ss.supplier_key
WHERE spo.actual_delivery IS NOT NULL
GROUP BY ss.supplier_key, ss.supplier_name, ss.country, ss.rating
HAVING COUNT(*) >= 5
ORDER BY on_time_pct DESC
""",
        "tables_needed": ["sc_purchase_orders", "sc_suppliers"],
    },
    {
        "task_id": "warehouse_017",
        "difficulty": "hard",
        "question": "Identify high-value customers (top 10% by lifetime value) who have had more than 3 support tickets in the last 6 months. Show their tier, total spend, number of tickets, and average satisfaction score.",
        "gold_sql": """
WITH customer_spend AS (
    SELECT customer_key, SUM(revenue) AS total_spend
    FROM fact_sales WHERE status = 'completed'
    GROUP BY customer_key
),
spend_threshold AS (
    SELECT total_spend AS threshold
    FROM customer_spend
    ORDER BY total_spend DESC
    LIMIT 1 OFFSET (SELECT COUNT(*) / 10 FROM customer_spend)
),
high_value AS (
    SELECT customer_key, total_spend
    FROM customer_spend
    WHERE total_spend >= (SELECT threshold FROM spend_threshold)
),
recent_tickets AS (
    SELECT
        customer_key,
        COUNT(*) AS ticket_count,
        AVG(satisfaction_score) AS avg_satisfaction
    FROM cs_tickets
    WHERE created_at >= date('2024-12-31', '-6 months')
    GROUP BY customer_key
    HAVING COUNT(*) > 3
)
SELECT
    dc.customer_key, dc.first_name, dc.last_name, dc.tier,
    hv.total_spend,
    rt.ticket_count,
    ROUND(rt.avg_satisfaction, 2) AS avg_satisfaction
FROM high_value hv
JOIN recent_tickets rt ON hv.customer_key = rt.customer_key
JOIN dim_customer dc ON hv.customer_key = dc.customer_key
ORDER BY hv.total_spend DESC
LIMIT 20
""",
        "tables_needed": ["fact_sales", "cs_tickets", "dim_customer"],
    },
    {
        "task_id": "warehouse_018",
        "difficulty": "medium",
        "question": "Show the rolling 7-day average of daily revenue for Q4 2024. Include the daily revenue and the 7-day moving average.",
        "gold_sql": """
WITH daily_revenue AS (
    SELECT
        dd.full_date,
        SUM(fs.revenue) AS daily_rev
    FROM fact_sales fs
    JOIN dim_date dd ON fs.date_key = dd.date_key
    WHERE dd.year = 2024 AND dd.quarter = 4
      AND fs.status = 'completed'
    GROUP BY dd.full_date
)
SELECT
    full_date,
    ROUND(daily_rev, 2) AS daily_revenue,
    ROUND(AVG(daily_rev) OVER (ORDER BY full_date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW), 2) AS rolling_7day_avg
FROM daily_revenue
ORDER BY full_date
""",
        "tables_needed": ["fact_sales", "dim_date"],
    },
    {
        "task_id": "warehouse_019",
        "difficulty": "hard",
        "question": "What's the revenue attribution across marketing touchpoints? For each touchpoint type (impression, click, form_fill, demo_request, purchase), show the total attributed revenue using each attribution model (first_touch, last_touch, linear, time_decay, position_based).",
        "gold_sql": """
SELECT
    touchpoint_type,
    attribution_model,
    COUNT(*) AS touchpoint_count,
    ROUND(SUM(revenue_attributed), 2) AS total_attributed_revenue,
    ROUND(AVG(revenue_attributed), 2) AS avg_attributed_revenue
FROM mktg_attribution
GROUP BY touchpoint_type, attribution_model
ORDER BY touchpoint_type, total_attributed_revenue DESC
""",
        "tables_needed": ["mktg_attribution"],
    },
    {
        "task_id": "warehouse_020",
        "difficulty": "hard",
        "question": "Calculate employee compensation vs performance. For each department, show the average salary, average performance rating, and identify if there's a correlation — specifically, show departments where the highest-paid quartile has lower average ratings than the lowest-paid quartile.",
        "gold_sql": """
WITH emp_quartiles AS (
    SELECT
        he.emp_key,
        hd.department_name,
        he.salary,
        NTILE(4) OVER (PARTITION BY he.department_key ORDER BY he.salary) AS salary_quartile
    FROM hr_employees he
    JOIN hr_departments hd ON he.department_key = hd.department_key
    WHERE he.employment_status = 'active'
),
quartile_perf AS (
    SELECT
        eq.department_name,
        eq.salary_quartile,
        ROUND(AVG(eq.salary), 2) AS avg_salary,
        ROUND(AVG(pr.overall_rating), 2) AS avg_rating,
        COUNT(DISTINCT eq.emp_key) AS employee_count
    FROM emp_quartiles eq
    JOIN hr_performance_reviews pr ON eq.emp_key = pr.emp_key
    GROUP BY eq.department_name, eq.salary_quartile
)
SELECT
    q1.department_name,
    q1.avg_salary AS q1_avg_salary, q1.avg_rating AS q1_avg_rating,
    q4.avg_salary AS q4_avg_salary, q4.avg_rating AS q4_avg_rating,
    ROUND(q4.avg_rating - q1.avg_rating, 2) AS rating_diff_q4_minus_q1
FROM quartile_perf q1
JOIN quartile_perf q4 ON q1.department_name = q4.department_name
WHERE q1.salary_quartile = 1 AND q4.salary_quartile = 4
  AND q4.avg_rating < q1.avg_rating
ORDER BY rating_diff_q4_minus_q1
""",
        "tables_needed": ["hr_employees", "hr_departments", "hr_performance_reviews"],
    },
]


def generate_warehouse_prompts(db_path: str, output_path: str, seed: int = 42):
    """Generate NL2SQL prompts with gold SQL for the warehouse benchmark."""
    tasks = []
    for gq in GOLD_QUERIES:
        tasks.append({
            "task_id": gq["task_id"],
            "difficulty": gq["difficulty"],
            "messages": [{"role": "user", "content": gq["question"]}],
            "gold_sql": gq["gold_sql"].strip(),
            "tables_needed": gq["tables_needed"],
        })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for t in tasks:
            f.write(json.dumps(t) + "\n")
    print(f"Generated {len(tasks)} warehouse NL2SQL prompts to {output_path}")


# ---------------------------------------------------------------------------
# 14. Result-Set Matching Reward
# ---------------------------------------------------------------------------

def evaluate_sql(db_path: str, predicted_sql: str, gold_sql: str,
                 order_matters: bool = False, tolerance: float = 0.01) -> dict:
    """Compare the result of predicted SQL against gold SQL.

    Returns a dict with:
      - score: float in [0, 1]
      - details: human-readable explanation
      - predicted_rows / gold_rows: row counts
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    try:
        gold_cursor = conn.execute(gold_sql)
        gold_cols = [d[0] for d in gold_cursor.description]
        gold_rows = [dict(r) for r in gold_cursor.fetchall()]
    except Exception as e:
        conn.close()
        return {"score": 0.0, "details": f"Gold SQL failed: {e}", "predicted_rows": 0, "gold_rows": 0}

    try:
        pred_cursor = conn.execute(predicted_sql)
        pred_cols = [d[0] for d in pred_cursor.description]
        pred_rows = [dict(r) for r in pred_cursor.fetchall()]
    except Exception as e:
        conn.close()
        return {"score": 0.0, "details": f"Predicted SQL failed: {e}",
                "predicted_rows": 0, "gold_rows": len(gold_rows)}

    conn.close()

    if not gold_rows and not pred_rows:
        return {"score": 1.0, "details": "Both returned empty results",
                "predicted_rows": 0, "gold_rows": 0}

    if not gold_rows or not pred_rows:
        return {"score": 0.0, "details": "One returned empty, other did not",
                "predicted_rows": len(pred_rows), "gold_rows": len(gold_rows)}

    def _normalize_row(row):
        """Normalize row values for comparison. Uses values only (ignores
        column names) so the comparison works across different aliases."""
        return tuple(
            round(v / tolerance) * tolerance if isinstance(v, float) else v
            for v in row.values()
        )

    if order_matters:
        matches = 0
        for g, p in zip(gold_rows, pred_rows):
            if _normalize_row(g) == _normalize_row(p):
                matches += 1
        score = matches / max(len(gold_rows), len(pred_rows))
    else:
        gold_set = set(_normalize_row(r) for r in gold_rows)
        pred_set = set(_normalize_row(r) for r in pred_rows)
        intersection = gold_set & pred_set
        union = gold_set | pred_set
        score = len(intersection) / len(union) if union else 1.0

    details = f"Predicted {len(pred_rows)} rows, gold {len(gold_rows)} rows. "
    if score == 1.0:
        details += "Exact match."
    elif score > 0.8:
        details += f"Close match (score={score:.2f})."
    elif score > 0.5:
        details += f"Partial match (score={score:.2f})."
    else:
        details += f"Poor match (score={score:.2f})."

    return {
        "score": round(score, 4),
        "details": details,
        "predicted_rows": len(pred_rows),
        "gold_rows": len(gold_rows),
    }


# ---------------------------------------------------------------------------
# 15. Schema Obfuscation — Enterprise Legacy Naming
# ---------------------------------------------------------------------------
# Real enterprise data warehouses use abbreviated, inconsistent naming
# conventions that make schema discovery genuinely hard. This obfuscation
# layer renames tables and columns to simulate that challenge.

import re

OBFUSCATE_TABLE_MAP = {
    # Core dimensions
    "dim_customer": "dw_cst_mstr",
    "dim_product": "dw_prd_mstr",
    "dim_store": "dw_loc_ref",
    "dim_date": "dw_cal_dim",
    "dim_channel": "dw_chnl_lkp",
    "dim_currency": "dw_ccy_ref",
    "dim_promotion": "dw_prmo_cfg",
    "dim_geography": "dw_geo_hier",
    # Fact tables
    "fact_sales": "trx_sls_dtl",
    "fact_returns": "trx_rtn_dtl",
    "fact_inventory_snapshot": "trx_inv_snap",
    "fact_web_traffic": "trx_web_evt",
    "fact_daily_sales_agg": "trx_sls_day_agg",
    "fact_monthly_customer_agg": "trx_cst_mth_agg",
    "fact_product_reviews": "trx_prd_rvw",
    # Finance
    "fin_chart_of_accounts": "gl_coa_mstr",
    "fin_journal_entries": "gl_jrnl_dtl",
    "fin_general_ledger": "gl_bal_dtl",
    "fin_invoices": "ap_inv_hdr",
    "fin_invoice_lines": "ap_inv_ln",
    "fin_payments": "ap_pmt_dtl",
    "fin_budgets": "fp_bdgt_dtl",
    "fin_cost_centers": "fp_cc_ref",
    "fin_accounts_receivable": "ar_aging_dtl",
    "fin_accounts_payable": "ap_vndr_aging",
    "fin_tax_rates": "tx_rate_ref",
    "fin_exchange_rates": "fx_rate_dtl",
    # HR
    "hr_employees": "hcm_wrk_mstr",
    "hr_departments": "hcm_org_ref",
    "hr_positions": "hcm_pos_ref",
    "hr_payroll": "hcm_pyrl_dtl",
    "hr_performance_reviews": "hcm_perf_rvw",
    "hr_compensation_history": "hcm_comp_hist",
    "hr_benefits_enrollment": "hcm_ben_enrl",
    "hr_benefit_plans": "hcm_ben_pln",
    "hr_time_off": "hcm_abs_dtl",
    "hr_training": "hcm_trng_dtl",
    # Marketing
    "mktg_campaigns": "crm_cmpgn_hdr",
    "mktg_leads": "crm_lead_dtl",
    "mktg_attribution": "crm_attr_dtl",
    "mktg_segments": "crm_seg_ref",
    "mktg_email_sends": "crm_eml_log",
    "mktg_ab_tests": "crm_ab_rslt",
    "mktg_content": "crm_cnt_dtl",
    "mktg_social_metrics": "crm_soc_met",
    # Supply chain
    "sc_suppliers": "scm_vndr_mstr",
    "sc_purchase_orders": "scm_po_hdr",
    "sc_po_lines": "scm_po_ln",
    "sc_warehouses": "scm_wh_ref",
    "sc_shipments": "scm_shp_dtl",
    "sc_carriers": "scm_crr_ref",
    "sc_inventory_movements": "scm_inv_mvmt",
    "sc_quality_inspections": "scm_qi_dtl",
    # Customer support
    "cs_tickets": "srm_case_dtl",
    "cs_agents": "srm_agt_ref",
    "cs_sla_definitions": "srm_sla_cfg",
    "cs_escalations": "srm_esc_dtl",
    "cs_feedback": "srm_fbk_dtl",
    "cs_knowledge_base": "srm_kb_ref",
    "cs_chat_sessions": "srm_chat_log",
    # Web analytics
    "wa_sessions": "web_sess_log",
    "wa_pageviews": "web_pv_log",
    "wa_events": "web_evt_log",
    "wa_conversions": "web_conv_dtl",
    "wa_experiments": "web_exp_rslt",
    "wa_search_queries": "web_srch_log",
    "wa_funnel_steps": "web_fnl_met",
    # Data engineering
    "de_etl_jobs": "ops_etl_job",
    "de_data_quality_rules": "ops_dq_rule",
    "de_data_quality_results": "ops_dq_rslt",
    "de_lineage": "ops_lnge_map",
    "de_schema_registry": "ops_schm_reg",
}

OBFUSCATE_COLUMN_MAP = {
    # Foreign keys — most critical for hiding join paths
    "customer_key": "cst_key",
    "customer_id": "cst_id",
    "product_key": "prd_key",
    "product_id": "prd_id",
    "store_key": "loc_key",
    "date_key": "cal_key",
    "channel_key": "chnl_key",
    "currency_key": "ccy_key",
    "promo_key": "prmo_key",
    "supplier_key": "vndr_key",
    "department_key": "org_key",
    "agent_key": "agt_key",
    "manager_key": "mgr_key",
    "campaign_id": "cmpgn_id",
    "invoice_id": "inv_id",
    "account_id": "acct_id",
    "warehouse_id": "wh_id",
    "carrier_id": "crr_id",
    "ticket_id": "case_id",
    "session_id": "sess_id",
    # Revenue / financial metrics
    "revenue": "net_amt",
    "cost_of_goods": "cogs_amt",
    "profit": "gm_amt",
    "unit_price": "unit_prc",
    "unit_cost": "unit_cst",
    "list_price": "lst_prc",
    "discount_amount": "disc_amt",
    "discount_pct": "disc_pct",
    "refund_amount": "rfnd_amt",
    "total_amount": "tot_amt",
    "restocking_fee": "rstk_fee",
    "quantity": "qty",
    "spend": "spnd_amt",
    "revenue_attributed": "attr_rev",
    "satisfaction_score": "csat_val",
    "resolution_time_min": "rslv_dur",
    "first_response_time_min": "frt_dur",
    "lifetime_value": "ltv_amt",
    "salary": "base_comp",
    "budget": "alloc_amt",
    "budget_amount": "bdgt_amt",
    "actual_amount": "actl_amt",
    "amount_due": "due_amt",
    "amount_paid": "paid_amt",
    # Dates
    "order_date": "txn_dt",
    "ship_date": "shp_dt",
    "delivery_date": "dlvr_dt",
    "signup_date": "reg_dt",
    "full_date": "cal_dt",
    "hire_date": "start_dt",
    "termination_date": "sep_dt",
    "created_at": "crt_ts",
    "resolved_at": "rslv_ts",
    "expected_delivery": "est_dlvr_dt",
    "actual_delivery": "act_dlvr_dt",
    "last_activity_date": "lst_actv_dt",
    "return_date": "rtn_dt",
    # Dimensions / categories
    "product_line": "prd_ln",
    "category": "catg_cd",
    "subcategory": "sub_catg",
    "status": "stat_cd",
    "channel_name": "chnl_nm",
    "department_name": "dept_nm",
    "campaign_name": "cmpgn_nm",
    "campaign_type": "cmpgn_type",
    "supplier_name": "vndr_nm",
    "employment_status": "emp_stat",
    "is_active": "actv_flg",
    "segment": "seg_cd",
    "tier": "tier_cd",
    "industry": "ind_cd",
    "acquisition_channel": "acq_src",
    "priority": "prty_cd",
    "reason_code": "rsn_cd",
    "touchpoint_type": "tp_type",
    "attribution_model": "attr_mdl",
    "device_type": "dvc_type",
    "payment_method": "pmt_mthd",
    "country": "cntry_cd",
    "rating": "rtng_val",
    # Date dimension columns
    "year": "cal_yr",
    "quarter": "cal_qtr",
    "month": "cal_mth",
    "fiscal_year": "fy",
    "fiscal_quarter": "fq",
    # Names
    "first_name": "frst_nm",
    "last_name": "lst_nm",
    "order_id": "txn_ref",
    "overall_rating": "ovrl_rtng",
    "days_outstanding": "days_os",
    "aging_bucket": "aging_bkt",
    "team": "grp_cd",
    "on_time_pct": "ot_pct",
}


def _build_full_table_map(conn):
    """Expand base table map to include all derivative tables (stg_, raw_,
    _history, _archive, audit_)."""
    all_names = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')").fetchall()}

    full_map = {}
    for old_base, new_base in OBFUSCATE_TABLE_MAP.items():
        if old_base in all_names:
            full_map[old_base] = new_base
        # Derivative prefixes
        for pfx in ("stg_", "raw_"):
            deriv = f"{pfx}{old_base}"
            if deriv in all_names:
                full_map[deriv] = f"{pfx}{new_base}"
        # Derivative suffixes
        for sfx, new_sfx in (("_history", "_hist"), ("_archive", "_arch")):
            deriv = f"{old_base}{sfx}"
            if deriv in all_names:
                full_map[deriv] = f"{new_base}{new_sfx}"
        # Audit prefix
        aud = f"audit_{old_base}"
        if aud in all_names:
            full_map[aud] = f"aud_{new_base}"

    return full_map


def _transform_sql(sql, table_map, column_map):
    """Replace table and column names in a SQL string.
    Longer names are replaced first to avoid partial matches."""
    result = sql
    for old, new in sorted(table_map.items(), key=lambda x: -len(x[0])):
        result = re.sub(r'\b' + re.escape(old) + r'\b', new, result)
    for old, new in sorted(column_map.items(), key=lambda x: -len(x[0])):
        result = re.sub(r'\b' + re.escape(old) + r'\b', new, result)
    return result


def obfuscate_warehouse(db_path: str):
    """Rename tables and columns to enterprise-legacy abbreviated names.
    Uses ALTER TABLE RENAME (SQLite 3.25+) to rename in-place."""
    conn = sqlite3.connect(db_path)

    full_table_map = _build_full_table_map(conn)
    print(f"Obfuscating {len(full_table_map)} tables, {len(OBFUSCATE_COLUMN_MAP)} column patterns...")

    # Phase 1: rename columns (before table renames so table names still match)
    all_tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    col_renames = 0
    for tname in all_tables:
        cols = conn.execute(f"PRAGMA table_info(\"{tname}\")").fetchall()
        for col in cols:
            col_name = col[1]
            if col_name in OBFUSCATE_COLUMN_MAP:
                new_col = OBFUSCATE_COLUMN_MAP[col_name]
                try:
                    conn.execute(f'ALTER TABLE "{tname}" RENAME COLUMN "{col_name}" TO "{new_col}"')
                    col_renames += 1
                except Exception:
                    pass
    conn.commit()
    print(f"  Renamed {col_renames} columns")

    # Phase 2: drop views (they reference old names and can't be altered)
    views = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='view'").fetchall()]
    for vname in views:
        try:
            conn.execute(f'DROP VIEW IF EXISTS "{vname}"')
        except Exception:
            pass
    conn.commit()

    # Phase 3: rename tables
    tbl_renames = 0
    for old_name, new_name in sorted(full_table_map.items(), key=lambda x: -len(x[0])):
        try:
            conn.execute(f'ALTER TABLE "{old_name}" RENAME TO "{new_name}"')
            tbl_renames += 1
        except Exception as e:
            print(f"  Warning: {old_name} -> {new_name}: {e}")
    conn.commit()
    print(f"  Renamed {tbl_renames} tables")

    # Phase 4: recreate placeholder views with obfuscated names
    view_names = [
        "v_mth_rev_sum", "v_cst_360", "v_prd_ctlg", "v_inv_avail",
        "v_open_cases", "v_wrk_dir", "v_pipe_sum", "v_mktg_roi",
        "v_vndr_card", "v_kpi_dash", "v_cst_hlth", "v_prd_margin",
        "v_sls_rgn", "v_sla_comp", "v_fnl_conv",
    ]
    for vn in view_names:
        try:
            conn.execute(f"CREATE VIEW {vn} AS SELECT 1 AS placeholder")
        except Exception:
            pass
    conn.commit()

    final_tables = conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'").fetchone()[0]
    final_views = conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='view'").fetchone()[0]
    conn.close()
    print(f"  Final: {final_tables} tables, {final_views} views")
    return full_table_map


def generate_obfuscated_prompts(db_path: str, output_path: str, table_map: dict, seed: int = 42):
    """Generate prompts with gold SQL transformed to use obfuscated names."""
    tasks = []
    for gq in GOLD_QUERIES:
        transformed_sql = _transform_sql(gq["gold_sql"], table_map, OBFUSCATE_COLUMN_MAP)
        transformed_tables = [table_map.get(t, t) for t in gq["tables_needed"]]
        tasks.append({
            "task_id": gq["task_id"],
            "difficulty": gq["difficulty"],
            "messages": [{"role": "user", "content": gq["question"]}],
            "gold_sql": transformed_sql.strip(),
            "tables_needed": transformed_tables,
        })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for t in tasks:
            f.write(json.dumps(t) + "\n")
    print(f"Generated {len(tasks)} obfuscated prompts to {output_path}")


# ---------------------------------------------------------------------------
# 16. CLI Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a 500+ table enterprise data warehouse for NL2SQL benchmarking."
    )
    parser.add_argument("--db-path", default="benchmark/data/enterprise_warehouse.db",
                        help="Output path for the SQLite database")
    parser.add_argument("--output", default="benchmark/data/nl2sql_warehouse_prompts.jsonl",
                        help="Output path for the prompts JSONL")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--obfuscate", action="store_true",
                        help="Obfuscate table/column names to enterprise-legacy style")
    parser.add_argument("--validate", action="store_true",
                        help="Validate gold SQL queries after generation")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.db_path), exist_ok=True)

    total = create_warehouse_db(args.db_path, seed=args.seed)

    if args.obfuscate:
        table_map = obfuscate_warehouse(args.db_path)
        generate_obfuscated_prompts(args.db_path, args.output, table_map, seed=args.seed)
    else:
        generate_warehouse_prompts(args.db_path, args.output, seed=args.seed)

    if args.validate:
        print("\nValidating gold SQL queries...")
        conn = sqlite3.connect(args.db_path)
        prompts = []
        with open(args.output) as f:
            for line in f:
                prompts.append(json.loads(line))
        for p in prompts:
            try:
                cursor = conn.execute(p["gold_sql"])
                rows = cursor.fetchall()
                print(f"  {p['task_id']}: OK ({len(rows)} rows)")
            except Exception as e:
                print(f"  {p['task_id']}: FAILED - {e}")
        conn.close()

    print(f"\nDone. Total objects in warehouse: {total}")
