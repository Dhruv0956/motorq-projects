import sqlite3
from pathlib import Path

import mysql.connector

from cleo.config import settings


DB_PATH = Path("cleo.db")


def init_sqlite():
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS todos (id INTEGER PRIMARY KEY AUTOINCREMENT, task TEXT NOT NULL, done INTEGER DEFAULT 0)"
        )


def add_todo(task):
    init_sqlite()
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute("INSERT INTO todos (task) VALUES (?)", (task,))
    return f"Added todo: {task}"


def list_todos():
    init_sqlite()
    with sqlite3.connect(DB_PATH) as connection:
        rows = connection.execute("SELECT id, task, done FROM todos ORDER BY id").fetchall()
    if not rows:
        return "Your to-do list is empty."
    return "\n".join(f"{row[0]}. {'[x]' if row[2] else '[ ]'} {row[1]}" for row in rows)


def mysql_connection():
    return mysql.connector.connect(
        host=settings.mysql_host,
        user=settings.mysql_user,
        password=settings.mysql_password,
        database=settings.mysql_database,
    )
