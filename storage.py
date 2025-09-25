"""
Shared CSV-based storage for student records.

This module is intentionally database-agnostic so that both the Flask app (app.py)
and a CLI (main.py) can read/write the same studentdetails.csv file.

Record shape keys:
- Name
- Moodle ID
- Email ID
- Parent's Email
- Contact Number
"""
from __future__ import annotations
import csv
import os
from typing import Dict, List, Tuple

STUDENTS_CSV = 'studentdetails.csv'
FIELDNAMES = ['Name', 'Moodle ID', 'Email ID', "Parent's Email", 'Contact Number']


def _ensure_file(path: str = STUDENTS_CSV) -> None:
    if not os.path.exists(path):
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()


def read_students(file_path: str = STUDENTS_CSV) -> List[Tuple[str, str, str, str, str]]:
    """Return list of tuples (Name, Moodle ID, Email ID, Parent's Email, Contact Number)."""
    students: List[Tuple[str, str, str, str, str]] = []
    if not os.path.exists(file_path):
        return students
    with open(file_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            students.append((
                row.get('Name', ''),
                str(row.get('Moodle ID', '')),
                row.get('Email ID', ''),
                row.get("Parent's Email", ''),
                row.get('Contact Number', '')
            ))
    return students


def upsert_student(record: Dict[str, str], file_path: str = STUDENTS_CSV) -> None:
    """Insert or update a student by Moodle ID."""
    _ensure_file(file_path)
    rows: List[Dict[str, str]] = []
    found = False
    with open(file_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            if str(r.get('Moodle ID', '')).strip() == str(record.get('Moodle ID', '')).strip():
                rows.append({k: record.get(k, '') for k in FIELDNAMES})
                found = True
            else:
                rows.append({k: r.get(k, '') for k in FIELDNAMES})
    if not found:
        rows.append({k: record.get(k, '') for k in FIELDNAMES})
    with open(file_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def delete_student(moodle_id: str, file_path: str = STUDENTS_CSV) -> None:
    """Delete a student by Moodle ID. No error if not found."""
    if not os.path.exists(file_path):
        return
    rows: List[Dict[str, str]] = []
    with open(file_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            if str(r.get('Moodle ID', '')).strip() != str(moodle_id).strip():
                rows.append({k: r.get(k, '') for k in FIELDNAMES})
    with open(file_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    # Quick sanity check when running this module directly.
    print('Students currently in CSV:')
    for rec in read_students():
        print(rec)