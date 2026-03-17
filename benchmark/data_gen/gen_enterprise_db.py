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
Generate a mock enterprise SQLite database for NL2SQL and data pipeline benchmark tasks.

Creates tables: customers, orders, order_items, products, inventory,
suppliers, shipments, returns, support_tickets, web_events, employees, departments.
Each table has realistic columns and ~1000-10000 rows of synthetic data.
"""

import argparse
import json
import os
import random
import sqlite3
from datetime import datetime, timedelta


def create_enterprise_db(db_path: str, seed: int = 42):
    rng = random.Random(seed)
    if os.path.exists(db_path):
        os.remove(db_path)
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    # Departments
    c.execute("""CREATE TABLE departments (
        dept_id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        budget REAL,
        location TEXT,
        created_at TEXT
    )""")
    depts = ["Engineering", "Sales", "Marketing", "Finance", "HR", "Operations", "Legal", "Support", "Product", "Data Science"]
    for i, d in enumerate(depts):
        c.execute("INSERT INTO departments VALUES (?, ?, ?, ?, ?)",
                  (i + 1, d, rng.uniform(100000, 5000000), rng.choice(["NYC", "SF", "Austin", "London", "Berlin"]),
                   (datetime(2020, 1, 1) + timedelta(days=rng.randint(0, 365))).isoformat()))

    # Employees
    c.execute("""CREATE TABLE employees (
        emp_id INTEGER PRIMARY KEY,
        first_name TEXT, last_name TEXT,
        email TEXT, dept_id INTEGER,
        title TEXT, salary REAL,
        hire_date TEXT, manager_id INTEGER,
        FOREIGN KEY (dept_id) REFERENCES departments(dept_id)
    )""")
    first_names = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Henry", "Iris", "Jack",
                   "Karen", "Leo", "Mia", "Noah", "Olivia", "Peter", "Quinn", "Rachel", "Sam", "Tara"]
    last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez"]
    titles = ["Engineer", "Senior Engineer", "Staff Engineer", "Manager", "Director", "VP", "Analyst", "Specialist"]
    for i in range(500):
        fn = rng.choice(first_names)
        ln = rng.choice(last_names)
        c.execute("INSERT INTO employees VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                  (i + 1, fn, ln, f"{fn.lower()}.{ln.lower()}{i}@company.com",
                   rng.randint(1, 10), rng.choice(titles), rng.uniform(50000, 250000),
                   (datetime(2018, 1, 1) + timedelta(days=rng.randint(0, 2000))).isoformat(),
                   rng.randint(1, max(1, i)) if i > 0 else None))

    # Products
    c.execute("""CREATE TABLE products (
        product_id INTEGER PRIMARY KEY,
        name TEXT, category TEXT, subcategory TEXT,
        price REAL, cost REAL,
        weight_kg REAL, supplier_id INTEGER,
        created_at TEXT, is_active INTEGER DEFAULT 1
    )""")
    categories = {
        "Electronics": ["Laptops", "Phones", "Tablets", "Accessories", "Audio"],
        "Clothing": ["Men", "Women", "Kids", "Shoes", "Accessories"],
        "Home": ["Furniture", "Kitchen", "Decor", "Garden", "Tools"],
        "Food": ["Snacks", "Beverages", "Organic", "Frozen", "Fresh"],
    }
    for i in range(200):
        cat = rng.choice(list(categories.keys()))
        subcat = rng.choice(categories[cat])
        price = round(rng.uniform(5, 500), 2)
        c.execute("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                  (i + 1, f"{subcat} Item {i}", cat, subcat, price, round(price * rng.uniform(0.3, 0.7), 2),
                   round(rng.uniform(0.1, 20), 2), rng.randint(1, 50),
                   (datetime(2020, 1, 1) + timedelta(days=rng.randint(0, 1000))).isoformat(),
                   1 if rng.random() > 0.1 else 0))

    # Customers
    c.execute("""CREATE TABLE customers (
        customer_id INTEGER PRIMARY KEY,
        first_name TEXT, last_name TEXT, email TEXT,
        country TEXT, state TEXT, city TEXT,
        signup_date TEXT, tier TEXT,
        lifetime_value REAL DEFAULT 0
    )""")
    countries = ["US", "UK", "DE", "FR", "CA", "AU", "JP", "BR"]
    tiers = ["Free", "Basic", "Premium", "Enterprise"]
    for i in range(5000):
        fn = rng.choice(first_names)
        ln = rng.choice(last_names)
        c.execute("INSERT INTO customers VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                  (i + 1, fn, ln, f"{fn.lower()}{i}@example.com",
                   rng.choice(countries), f"State-{rng.randint(1,50)}", f"City-{rng.randint(1,100)}",
                   (datetime(2019, 1, 1) + timedelta(days=rng.randint(0, 2000))).isoformat(),
                   rng.choice(tiers), round(rng.uniform(0, 10000), 2)))

    # Suppliers
    c.execute("""CREATE TABLE suppliers (
        supplier_id INTEGER PRIMARY KEY,
        name TEXT, country TEXT, contact_email TEXT,
        rating REAL, contract_start TEXT, contract_end TEXT
    )""")
    for i in range(50):
        start = datetime(2020, 1, 1) + timedelta(days=rng.randint(0, 1000))
        c.execute("INSERT INTO suppliers VALUES (?, ?, ?, ?, ?, ?, ?)",
                  (i + 1, f"Supplier {i}", rng.choice(countries),
                   f"contact@supplier{i}.com", round(rng.uniform(1, 5), 1),
                   start.isoformat(), (start + timedelta(days=rng.randint(365, 1095))).isoformat()))

    # Orders
    c.execute("""CREATE TABLE orders (
        order_id INTEGER PRIMARY KEY,
        customer_id INTEGER, order_date TEXT,
        status TEXT, total_amount REAL,
        shipping_address TEXT, payment_method TEXT,
        FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
    )""")
    statuses = ["pending", "processing", "shipped", "delivered", "cancelled", "returned"]
    payment_methods = ["credit_card", "debit_card", "paypal", "bank_transfer", "crypto"]
    for i in range(10000):
        c.execute("INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?, ?)",
                  (i + 1, rng.randint(1, 5000),
                   (datetime(2022, 1, 1) + timedelta(days=rng.randint(0, 730))).isoformat(),
                   rng.choice(statuses), round(rng.uniform(10, 2000), 2),
                   f"{rng.randint(1,999)} Main St, City-{rng.randint(1,100)}",
                   rng.choice(payment_methods)))

    # Order Items
    c.execute("""CREATE TABLE order_items (
        item_id INTEGER PRIMARY KEY,
        order_id INTEGER, product_id INTEGER,
        quantity INTEGER, unit_price REAL, discount REAL DEFAULT 0,
        FOREIGN KEY (order_id) REFERENCES orders(order_id),
        FOREIGN KEY (product_id) REFERENCES products(product_id)
    )""")
    item_id = 1
    for oid in range(1, 10001):
        for _ in range(rng.randint(1, 5)):
            pid = rng.randint(1, 200)
            qty = rng.randint(1, 10)
            price = round(rng.uniform(5, 500), 2)
            c.execute("INSERT INTO order_items VALUES (?, ?, ?, ?, ?, ?)",
                      (item_id, oid, pid, qty, price, round(rng.uniform(0, 0.3), 2)))
            item_id += 1

    # Inventory
    c.execute("""CREATE TABLE inventory (
        product_id INTEGER PRIMARY KEY,
        warehouse TEXT, quantity INTEGER,
        reorder_point INTEGER, last_restocked TEXT,
        FOREIGN KEY (product_id) REFERENCES products(product_id)
    )""")
    warehouses = ["US-East", "US-West", "EU-Central", "APAC"]
    for i in range(1, 201):
        c.execute("INSERT INTO inventory VALUES (?, ?, ?, ?, ?)",
                  (i, rng.choice(warehouses), rng.randint(0, 1000),
                   rng.randint(10, 100),
                   (datetime(2023, 1, 1) + timedelta(days=rng.randint(0, 365))).isoformat()))

    # Support Tickets
    c.execute("""CREATE TABLE support_tickets (
        ticket_id INTEGER PRIMARY KEY,
        customer_id INTEGER, created_at TEXT,
        resolved_at TEXT, category TEXT,
        priority TEXT, status TEXT, description TEXT,
        FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
    )""")
    ticket_cats = ["billing", "technical", "shipping", "product_quality", "account", "other"]
    priorities = ["low", "medium", "high", "critical"]
    for i in range(3000):
        created = datetime(2023, 1, 1) + timedelta(days=rng.randint(0, 365))
        resolved = created + timedelta(hours=rng.randint(1, 720)) if rng.random() > 0.2 else None
        c.execute("INSERT INTO support_tickets VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                  (i + 1, rng.randint(1, 5000), created.isoformat(),
                   resolved.isoformat() if resolved else None,
                   rng.choice(ticket_cats), rng.choice(priorities),
                   "resolved" if resolved else rng.choice(["open", "in_progress"]),
                   f"Issue with {rng.choice(ticket_cats)}: customer reports {rng.choice(['error', 'delay', 'missing item', 'wrong item', 'access denied'])}"))

    # Web Events (clickstream)
    c.execute("""CREATE TABLE web_events (
        event_id INTEGER PRIMARY KEY,
        customer_id INTEGER, session_id TEXT,
        event_type TEXT, page_url TEXT,
        timestamp TEXT, device TEXT,
        FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
    )""")
    event_types = ["page_view", "click", "add_to_cart", "remove_from_cart", "checkout_start", "purchase", "search"]
    devices = ["desktop", "mobile", "tablet"]
    for i in range(20000):
        c.execute("INSERT INTO web_events VALUES (?, ?, ?, ?, ?, ?, ?)",
                  (i + 1, rng.randint(1, 5000), f"sess-{rng.randint(1, 10000):06d}",
                   rng.choice(event_types), f"/page/{rng.choice(['home', 'product', 'cart', 'checkout', 'search', 'category'])}/{rng.randint(1,100)}",
                   (datetime(2023, 6, 1) + timedelta(seconds=rng.randint(0, 15552000))).isoformat(),
                   rng.choice(devices)))

    # Returns
    c.execute("""CREATE TABLE returns (
        return_id INTEGER PRIMARY KEY,
        order_id INTEGER, reason TEXT,
        refund_amount REAL, return_date TEXT,
        status TEXT,
        FOREIGN KEY (order_id) REFERENCES orders(order_id)
    )""")
    return_reasons = ["defective", "wrong_item", "not_as_described", "changed_mind", "too_late", "damaged_in_shipping"]
    for i in range(1500):
        c.execute("INSERT INTO returns VALUES (?, ?, ?, ?, ?, ?)",
                  (i + 1, rng.randint(1, 10000), rng.choice(return_reasons),
                   round(rng.uniform(5, 500), 2),
                   (datetime(2023, 1, 1) + timedelta(days=rng.randint(0, 365))).isoformat(),
                   rng.choice(["pending", "approved", "rejected", "refunded"])))

    # Create some useful views
    c.execute("""CREATE VIEW customer_order_summary AS
        SELECT c.customer_id, c.first_name, c.last_name, c.tier,
               COUNT(o.order_id) as total_orders,
               COALESCE(SUM(o.total_amount), 0) as total_spent,
               MAX(o.order_date) as last_order_date
        FROM customers c LEFT JOIN orders o ON c.customer_id = o.customer_id
        GROUP BY c.customer_id""")

    c.execute("""CREATE VIEW product_performance AS
        SELECT p.product_id, p.name, p.category,
               COUNT(oi.item_id) as times_ordered,
               SUM(oi.quantity) as total_qty_sold,
               SUM(oi.quantity * oi.unit_price * (1 - oi.discount)) as revenue
        FROM products p LEFT JOIN order_items oi ON p.product_id = oi.product_id
        GROUP BY p.product_id""")

    # Create indexes
    c.execute("CREATE INDEX idx_orders_customer ON orders(customer_id)")
    c.execute("CREATE INDEX idx_orders_date ON orders(order_date)")
    c.execute("CREATE INDEX idx_order_items_order ON order_items(order_id)")
    c.execute("CREATE INDEX idx_web_events_customer ON web_events(customer_id)")
    c.execute("CREATE INDEX idx_web_events_session ON web_events(session_id)")
    c.execute("CREATE INDEX idx_support_customer ON support_tickets(customer_id)")

    conn.commit()
    conn.close()
    print(f"Created enterprise database at {db_path}")

    # Print table stats
    conn = sqlite3.connect(db_path)
    tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    for (t,) in tables:
        count = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        print(f"  {t}: {count} rows")
    conn.close()


def generate_nl2sql_prompts(db_path: str, output_path: str, n_prompts: int = 50, seed: int = 42):
    """Generate NL2SQL task prompts from the enterprise database."""
    rng = random.Random(seed)

    # Complex analytical questions
    questions = [
        "What are the top 10 customers by total spending, and what tier are they in?",
        "Show me the monthly revenue trend for the last 12 months, broken down by product category.",
        "Which products have a return rate higher than 15%, and what are the most common return reasons?",
        "Find customers who signed up in 2023 but haven't placed any orders yet.",
        "What's the average resolution time for support tickets by priority level and category?",
        "Calculate the conversion rate from web_events (page_view to purchase) by device type.",
        "Which suppliers have the highest-rated products and what's their average delivery time?",
        "Find orders where the total amount doesn't match the sum of order items (after discounts).",
        "What's the customer lifetime value distribution across different countries and tiers?",
        "Show me the inventory items that are below reorder point and their current stock levels.",
        "Analyze the correlation between support ticket volume and return rates by product category.",
        "Which employees manage the most people, and what's their department's budget per employee?",
        "Find the most common customer journey paths from web_events (sequence of page_urls per session).",
        "Calculate week-over-week growth in new customer signups by country for the last 3 months.",
        "What products are frequently bought together (appear in the same order)?",
        "Show departments that are over budget based on employee salaries vs department budget.",
        "Find customers with declining order frequency over the last 4 quarters.",
        "What's the average time between a customer's first and second order?",
        "Identify potential fraud: orders with unusually high amounts from new customers.",
        "Calculate the Net Promoter Score proxy from support ticket resolution rates by customer tier.",
        "Which product categories have the highest margin (price vs cost) and highest volume?",
        "Show the distribution of order values by payment method and customer country.",
        "Find sessions from web_events where users added items to cart but didn't complete checkout.",
        "What's the restock frequency needed for each warehouse based on order volume trends?",
        "Compare the performance of products launched in Q1 vs Q2 of 2023.",
        "Identify customers who have both high spending AND high support ticket counts.",
        "What's the average order value for each day of the week?",
        "Find products where the inventory quantity is sufficient but the product is marked inactive.",
        "Calculate customer churn rate by tier (customers who haven't ordered in 6+ months).",
        "Show the full chain: supplier -> product -> order_items -> orders -> returns for defective items.",
        "What percentage of orders from each country are shipped vs delivered vs cancelled?",
        "Find employees whose salary is more than 2 standard deviations above their department average.",
        "Which product subcategories have seen the biggest month-over-month growth?",
        "Calculate the rolling 7-day average of daily orders and identify peak periods.",
        "Find customers who have ordered from every product category.",
        "What's the average number of items per order by customer tier?",
        "Show web sessions that last more than 30 minutes (based on event timestamps).",
        "Find orders that were placed within 24 hours of a support ticket being opened.",
        "What's the correlation between customer signup recency and average order value?",
        "Identify the top 5 product pairs that generate the most combined revenue when in same order.",
        "Calculate the fill rate (orders fulfilled from inventory vs total orders) by warehouse.",
        "Show the trend of average discount given over time - is it increasing?",
        "Find departments where average employee tenure exceeds 3 years.",
        "What's the refund amount as a percentage of total revenue by product category?",
        "Identify web_events anomalies: sessions with unusually high event counts.",
        "Calculate customer acquisition cost proxy by tier (using marketing department budget / new customers).",
        "Show monthly active customers (at least one order) vs total customers over time.",
        "Find products that have never been ordered.",
        "What's the average time from order creation to delivery by shipping address state?",
        "Build a cohort analysis: retention rates for customers by their signup quarter.",
    ]

    tasks = []
    for i in range(n_prompts):
        q = questions[i % len(questions)]
        # Vary the question slightly
        if rng.random() > 0.5:
            q = q + f" Focus on data from {rng.choice(['2023', '2022', 'the last quarter'])}."
        if rng.random() > 0.7:
            q = "As a data analyst, " + q[0].lower() + q[1:]

        tasks.append({
            "task_id": f"nl2sql_{i:03d}",
            "messages": [{"role": "user", "content": q}],
        })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for t in tasks:
            f.write(json.dumps(t) + "\n")
    print(f"Generated {len(tasks)} NL2SQL prompts to {output_path}")


def generate_pipeline_debug_prompts(db_path: str, output_path: str, n_prompts: int = 50, seed: int = 42):
    """Generate data pipeline debugging task prompts."""
    rng = random.Random(seed)

    scenarios = [
        "The daily revenue report shows a 50% drop yesterday. Investigate whether this is a data quality issue or a real business decline. Check orders, web_events, and payment methods.",
        "The ETL pipeline for the customer_order_summary view is producing NULL values for some customers. Debug the join logic and check for data integrity issues.",
        "We're seeing duplicate entries in the web_events table. Write queries to identify duplicates, assess impact, and write a deduplication query.",
        "The inventory reorder alerts are not firing correctly. Some products are out of stock despite having reorder_point set. Debug the inventory management logic.",
        "Support ticket resolution time metrics jumped 3x last week. Investigate if it's a real slowdown or a data anomaly (e.g., timezone issues, null resolved_at).",
        "Customer tier assignments seem wrong - some high-spending customers are on the Free tier. Audit the tier logic and propose corrections.",
        "The order_items total doesn't match the orders.total_amount for about 5% of orders. Find these mismatches and diagnose whether it's a discount calculation bug.",
        "We discovered that some employee.manager_id values point to non-existent employees. Find these orphaned references and assess the impact on org chart queries.",
        "The product_performance view is running slowly. Analyze the query plan, suggest indexes, and optimize the view definition.",
        "Web analytics show impossible session durations (e.g., 0 seconds with 50 events). Identify these anomalies and determine if they're bots or data quality issues.",
        "The returns data seems to have future dates. Investigate temporal consistency across all date columns in the database.",
        "Shipping address data quality is poor - many orders have malformed addresses. Write queries to identify and categorize the issues.",
        "The conversion funnel metrics don't add up: more purchases than checkout_starts. Debug the web_events data pipeline.",
        "Customer email addresses have duplicates with different customer_ids. Find these, determine which is the primary account, and assess order history impact.",
        "The department budget vs actual salary spend is being calculated wrong in reports. Debug the aggregation logic.",
    ]

    tasks = []
    for i in range(n_prompts):
        scenario = scenarios[i % len(scenarios)]
        tasks.append({
            "task_id": f"pipeline_debug_{i:03d}",
            "messages": [{"role": "user", "content": scenario}],
        })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for t in tasks:
            f.write(json.dumps(t) + "\n")
    print(f"Generated {len(tasks)} pipeline debug prompts to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", default="benchmark/data/enterprise.db")
    parser.add_argument("--nl2sql-output", default="benchmark/data/nl2sql_prompts.jsonl")
    parser.add_argument("--pipeline-output", default="benchmark/data/pipeline_debug_prompts.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.db_path), exist_ok=True)
    create_enterprise_db(args.db_path, seed=args.seed)
    generate_nl2sql_prompts(args.db_path, args.nl2sql_output, seed=args.seed)
    generate_pipeline_debug_prompts(args.db_path, args.pipeline_output, seed=args.seed)
