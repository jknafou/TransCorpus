import sqlite3
import time
import random
from pathlib import Path
import psutil, os


def get_unique_process_id():
    pid = os.getpid()
    p = psutil.Process(pid)
    start_time = p.create_time()
    unique_id = f"{pid}_{start_time}"
    return unique_id


def is_process_alive(unique_id: str) -> bool:
    pid, start_time = unique_id.split("_")
    pid = int(pid)
    start_time = float(start_time)
    try:
        process = psutil.Process(pid)
        return (
            process.is_running()
            and process.create_time() == start_time
            and process.status() != psutil.STATUS_ZOMBIE
        )
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False


class TranslationCheckpointing:
    def __init__(self, db_path: Path, num_splits: int):
        self.db_path = db_path
        self._init_db()
        self.add_splits(num_splits)

    def _init_db(self):
        with self._connect() as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS splits (
                split_number INTEGER PRIMARY KEY,
                unique_id TEXT,
                status TEXT CHECK(status IN ('pending', 'in_progress', 'completed')),
                completed_stage INTEGER DEFAULT 0,
                claimed_at REAL,
                completed_at REAL
            )""")
            conn.commit()

    def _connect(self):
        return sqlite3.connect(self.db_path, timeout=30)

    def add_splits(self, num_splits: int):
        with self._connect() as conn:
            conn.executemany(
                "INSERT OR IGNORE INTO splits (split_number, status) VALUES (?, ?)",
                [(i, "pending") for i in range(1, num_splits + 1)],
            )
            conn.commit()

    def update_stage(self, split_number: int, stage: int):
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE splits SET completed_stage = ?
                WHERE split_number = ?
                """,
                (stage, split_number),
            )
            conn.commit()

    def get_stage(self, split_number: int) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                """
                SELECT completed_stage FROM splits
                WHERE split_number = ?
                """,
                (split_number,),
            )
            result = cursor.fetchone()
            return result[0] if result else 0

    def claim_next_split(self, max_retries=5) -> int | None:
        """Claim next split after verifying active processes"""
        unique_id = get_unique_process_id()
        for retry in range(max_retries):
            try:
                with self._connect() as conn:
                    cursor = conn.cursor()
                    cursor.execute("BEGIN IMMEDIATE")
                    # First: Check and reset stale in_progress splits
                    cursor.execute("""
                        SELECT split_number, unique_id FROM splits
                        WHERE status = 'in_progress'
                    """)
                    for split_number, stored_id in cursor.fetchall():
                        if not is_process_alive(stored_id):
                            cursor.execute(
                                """
                                UPDATE splits SET
                                    status = 'pending',
                                    unique_id = NULL,
                                    claimed_at = NULL
                                WHERE split_number = ?
                            """,
                                (split_number,),
                            )
                    # Now find next pending split
                    cursor.execute("""
                        SELECT split_number FROM splits
                        WHERE status = 'pending'
                        ORDER BY split_number
                        LIMIT 1
                    """)
                    result = cursor.fetchone()
                    if not result:
                        return None  # No splits left
                    split_number = result[0]
                    # Claim the split
                    cursor.execute(
                        """
                        UPDATE splits SET
                            status = 'in_progress',
                            unique_id = ?,
                            claimed_at = ?
                        WHERE split_number = ?
                    """,
                        (unique_id, time.time(), split_number),
                    )
                    conn.commit()
                    return split_number
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e) and retry < max_retries - 1:
                    sleep_time = random.uniform(0.1, 0.5) * (2**retry)
                    time.sleep(sleep_time)
                    continue
                raise
        return None

    def complete_split(self, split_number: int):
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE splits SET
                    status = 'completed',
                    completed_at = ?
                WHERE split_number = ?
            """,
                (time.time(), split_number),
            )
            conn.commit()

    def get_len_uncompleted_splits(self) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                """
                SELECT COUNT(*) FROM splits
                WHERE status != 'completed'
            """
            )
            result = cursor.fetchone()
        return result[0] if result else 0

    def get_status(self, split_number: int) -> dict | None:
        with self._connect() as conn:
            cursor = conn.execute(
                """
                SELECT status, unique_id, claimed_at, completed_at
                FROM splits
                WHERE split_number = ?
            """,
                (split_number,),
            )
            result = cursor.fetchone()
        return (
            {
                "status": result[0],
                "unique_id": result[1],
                "claimed_at": result[2],
                "completed_at": result[3],
            }
            if result
            else None
        )


# if __name__ == "__main__":
#     db_path = Path("test_splits.db")
#     split_db = TranslationCheckpointing(db_path, 10)
#     # split_db.add_splits(10)
#
#     # Example usage
#     print("Uncompleted splits count:", split_db.get_len_uncompleted_splits())
#     split_number = split_db.claim_next_split()
#     if split_number:
#         print(f"Claimed split number: {split_number}")
#         # Simulate work
#         time.sleep(2)
#         split_db.complete_split(split_number)
#         print(f"Completed split number: {split_number}")
#     else:
#         print("No splits available.")
#
#     print("Uncompleted splits count:", split_db.get_len_uncompleted_splits())
