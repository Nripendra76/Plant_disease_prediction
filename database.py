"""
PlantGuard AI — SQLite Database Module
Handles user authentication, prediction history, and admin statistics.
"""

import sqlite3
import hashlib
import os
import secrets
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plantguard.db")


def get_connection():
    """Get a database connection with row factory."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    """Initialize database tables and create default admin if not exists."""
    conn = get_connection()
    cursor = conn.cursor()

    # Users table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            role TEXT DEFAULT 'user' CHECK(role IN ('user', 'admin')),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Predictions table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            image_name TEXT,
            predicted_class TEXT NOT NULL,
            plant_name TEXT,
            disease_name TEXT,
            confidence REAL NOT NULL,
            is_healthy INTEGER DEFAULT 0,
            severity TEXT,
            disease_type TEXT,
            top3_predictions TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    """)

    conn.commit()

    # Create default admin account if not exists
    admin_exists = cursor.execute(
        "SELECT 1 FROM users WHERE username = ?", ("admin",)
    ).fetchone()

    if not admin_exists:
        salt = secrets.token_hex(16)
        password_hash = hashlib.sha256(("admin123" + salt).encode()).hexdigest()
        cursor.execute(
            "INSERT INTO users (username, email, password_hash, salt, role) VALUES (?, ?, ?, ?, ?)",
            ("admin", "admin@plantguard.ai", password_hash, salt, "admin")
        )
        conn.commit()

    conn.close()


def hash_password(password: str, salt: str = None) -> Tuple[str, str]:
    """Hash a password with salt. Returns (hash, salt)."""
    if salt is None:
        salt = secrets.token_hex(16)
    password_hash = hashlib.sha256((password + salt).encode()).hexdigest()
    return password_hash, salt


def create_user(username: str, email: str, password: str, role: str = "user") -> Tuple[bool, str]:
    """Create a new user. Returns (success, message)."""
    if len(username) < 3:
        return False, "Username must be at least 3 characters."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."
    if "@" not in email or "." not in email:
        return False, "Please enter a valid email address."

    conn = get_connection()
    try:
        password_hash, salt = hash_password(password)
        conn.execute(
            "INSERT INTO users (username, email, password_hash, salt, role) VALUES (?, ?, ?, ?, ?)",
            (username.strip(), email.strip().lower(), password_hash, salt, role)
        )
        conn.commit()
        return True, "Account created successfully!"
    except sqlite3.IntegrityError as e:
        if "username" in str(e):
            return False, "Username already exists."
        elif "email" in str(e):
            return False, "Email already registered."
        return False, "Registration failed."
    finally:
        conn.close()


def authenticate(username: str, password: str) -> Optional[Dict[str, Any]]:
    """Authenticate a user. Returns user dict or None."""
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM users WHERE username = ?", (username.strip(),)
    ).fetchone()
    conn.close()

    if row is None:
        return None

    password_hash, _ = hash_password(password, row["salt"])
    if password_hash == row["password_hash"]:
        return {
            "id": row["id"],
            "username": row["username"],
            "email": row["email"],
            "role": row["role"],
            "created_at": row["created_at"]
        }
    return None


def save_prediction(
    user_id: int,
    image_name: str,
    predicted_class: str,
    plant_name: str,
    disease_name: str,
    confidence: float,
    is_healthy: bool,
    severity: str,
    disease_type: str,
    top3_predictions: str = ""
):
    """Save a prediction to the database."""
    conn = get_connection()
    conn.execute(
        """INSERT INTO predictions
        (user_id, image_name, predicted_class, plant_name, disease_name,
         confidence, is_healthy, severity, disease_type, top3_predictions)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (user_id, image_name, predicted_class, plant_name, disease_name,
         confidence, int(is_healthy), severity, disease_type, top3_predictions)
    )
    conn.commit()
    conn.close()


def get_user_history(user_id: int, limit: int = 50) -> List[Dict]:
    """Get prediction history for a user."""
    conn = get_connection()
    rows = conn.execute(
        """SELECT * FROM predictions WHERE user_id = ?
           ORDER BY timestamp DESC LIMIT ?""",
        (user_id, limit)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_user_stats(user_id: int) -> Dict[str, Any]:
    """Get statistics for a user."""
    conn = get_connection()

    total = conn.execute(
        "SELECT COUNT(*) as cnt FROM predictions WHERE user_id = ?", (user_id,)
    ).fetchone()["cnt"]

    healthy = conn.execute(
        "SELECT COUNT(*) as cnt FROM predictions WHERE user_id = ? AND is_healthy = 1", (user_id,)
    ).fetchone()["cnt"]

    most_common = conn.execute(
        """SELECT predicted_class, COUNT(*) as cnt FROM predictions
           WHERE user_id = ? AND is_healthy = 0
           GROUP BY predicted_class ORDER BY cnt DESC LIMIT 1""",
        (user_id,)
    ).fetchone()

    conn.close()

    return {
        "total_scans": total,
        "healthy_count": healthy,
        "diseased_count": total - healthy,
        "healthy_ratio": (healthy / total * 100) if total > 0 else 0,
        "most_common_disease": dict(most_common)["predicted_class"] if most_common else "N/A"
    }


# ═══════════════════════════════════════
#  ADMIN FUNCTIONS
# ═══════════════════════════════════════

def get_all_users() -> List[Dict]:
    """Get all users with their scan counts."""
    conn = get_connection()
    rows = conn.execute("""
        SELECT u.id, u.username, u.email, u.role, u.created_at,
               COUNT(p.id) as scan_count
        FROM users u
        LEFT JOIN predictions p ON u.id = p.user_id
        GROUP BY u.id
        ORDER BY u.created_at DESC
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_platform_stats() -> Dict[str, Any]:
    """Get overall platform statistics."""
    conn = get_connection()

    total_users = conn.execute("SELECT COUNT(*) as cnt FROM users").fetchone()["cnt"]
    total_preds = conn.execute("SELECT COUNT(*) as cnt FROM predictions").fetchone()["cnt"]
    total_healthy = conn.execute("SELECT COUNT(*) as cnt FROM predictions WHERE is_healthy = 1").fetchone()["cnt"]

    disease_dist = conn.execute("""
        SELECT predicted_class, COUNT(*) as cnt FROM predictions
        WHERE is_healthy = 0
        GROUP BY predicted_class ORDER BY cnt DESC LIMIT 10
    """).fetchall()

    top_users = conn.execute("""
        SELECT u.username, COUNT(p.id) as scan_count
        FROM users u JOIN predictions p ON u.id = p.user_id
        GROUP BY u.id ORDER BY scan_count DESC LIMIT 5
    """).fetchall()

    recent_preds = conn.execute("""
        SELECT p.*, u.username FROM predictions p
        JOIN users u ON p.user_id = u.id
        ORDER BY p.timestamp DESC LIMIT 10
    """).fetchall()

    conn.close()

    return {
        "total_users": total_users,
        "total_predictions": total_preds,
        "total_healthy": total_healthy,
        "total_diseased": total_preds - total_healthy,
        "disease_distribution": [dict(r) for r in disease_dist],
        "top_users": [dict(r) for r in top_users],
        "recent_predictions": [dict(r) for r in recent_preds],
    }


def delete_user(user_id: int) -> bool:
    """Delete a user and their predictions."""
    conn = get_connection()
    conn.execute("DELETE FROM predictions WHERE user_id = ?", (user_id,))
    conn.execute("DELETE FROM users WHERE id = ? AND role != 'admin'", (user_id,))
    conn.commit()
    conn.close()
    return True


def clear_all_predictions():
    """Clear all prediction records."""
    conn = get_connection()
    conn.execute("DELETE FROM predictions")
    conn.commit()
    conn.close()


# Initialize database on module import
init_db()
