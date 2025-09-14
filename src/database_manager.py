"""
Updated Database Manager for Face Recognition System with HR Management
Handles SQLite database operations for storing face embeddings, person information, HR users, and interview schedules
"""

import sqlite3
import numpy as np
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import os


class DatabaseManager:
    def __init__(self, db_path: str = "face_recognition.db"):
        """
        Initialize Database Manager

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Initialize database tables"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Create people table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS people (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL UNIQUE,
                        age INTEGER,
                        major TEXT,
                        school TEXT,
                        meeting_room TEXT,
                        meeting_time TEXT,
                        phone TEXT,
                        email TEXT,
                        position_applied TEXT,
                        interview_status TEXT DEFAULT 'scheduled',
                        notes TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        active BOOLEAN DEFAULT TRUE
                    )
                ''')

                # Create face_embeddings table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS face_embeddings (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        person_id INTEGER NOT NULL,
                        embedding BLOB NOT NULL,
                        faces_processed INTEGER DEFAULT 1,
                        embedding_norm REAL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (person_id) REFERENCES people (id) ON DELETE CASCADE
                    )
                ''')

                # Create hr_users table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS hr_users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username TEXT NOT NULL UNIQUE,
                        password_hash TEXT NOT NULL,
                        full_name TEXT NOT NULL,
                        email TEXT,
                        role TEXT DEFAULT 'hr',
                        last_login TIMESTAMP,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        active BOOLEAN DEFAULT TRUE
                    )
                ''')

                # Create interview_schedules table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS interview_schedules (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        person_id INTEGER NOT NULL,
                        hr_user_id INTEGER NOT NULL,
                        interview_date TIMESTAMP NOT NULL,
                        interview_room TEXT NOT NULL,
                        interview_type TEXT DEFAULT 'technical',
                        duration_minutes INTEGER DEFAULT 60,
                        status TEXT DEFAULT 'scheduled',
                        notes TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (person_id) REFERENCES people (id) ON DELETE CASCADE,
                        FOREIGN KEY (hr_user_id) REFERENCES hr_users (id)
                    )
                ''')

                # Create indexes for better performance
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_people_name ON people (name)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_embeddings_person ON face_embeddings (person_id)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_hr_username ON hr_users (username)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_interviews_date ON interview_schedules (interview_date)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_interviews_person ON interview_schedules (person_id)')

                # Create default HR user if not exists
                cursor.execute('SELECT COUNT(*) FROM hr_users')
                if cursor.fetchone()[0] == 0:
                    self.create_default_hr_user(cursor)

                conn.commit()
                print("Database initialized successfully")

        except Exception as e:
            print(f"Error initializing database: {e}")
            raise

    def create_default_hr_user(self, cursor):
        """Create default HR user"""
        default_password = "admin123"
        password_hash = hashlib.sha256(default_password.encode()).hexdigest()

        cursor.execute('''
            INSERT INTO hr_users (username, password_hash, full_name, email, role)
            VALUES (?, ?, ?, ?, ?)
        ''', ('admin', password_hash, 'HR Administrator', 'admin@company.com', 'admin'))

        print("Created default HR user: admin / admin123")

    def hash_password(self, password: str) -> str:
        """Hash password using SHA256"""
        return hashlib.sha256(password.encode()).hexdigest()

    def verify_hr_user(self, username: str, password: str) -> Optional[Dict]:
        """Verify HR user credentials"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                password_hash = self.hash_password(password)

                cursor.execute('''
                    SELECT id, username, full_name, email, role, last_login
                    FROM hr_users 
                    WHERE username = ? AND password_hash = ? AND active = TRUE
                ''', (username, password_hash))

                result = cursor.fetchone()
                if result:
                    # Update last login
                    cursor.execute('''
                        UPDATE hr_users SET last_login = CURRENT_TIMESTAMP 
                        WHERE id = ?
                    ''', (result[0],))
                    conn.commit()

                    return {
                        'id': result[0],
                        'username': result[1],
                        'full_name': result[2],
                        'email': result[3],
                        'role': result[4],
                        'last_login': result[5]
                    }
                return None

        except Exception as e:
            print(f"Error verifying HR user: {e}")
            return None

    def add_person(self, name: str, age: int = None, major: str = None,
                   school: str = None, meeting_room: str = None,
                   meeting_time: str = None, phone: str = None,
                   email: str = None, position_applied: str = None,
                   interview_status: str = 'scheduled', notes: str = None) -> int:
        """Add a new person to the database with extended fields"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute('''
                    INSERT INTO people (name, age, major, school, meeting_room, meeting_time,
                                      phone, email, position_applied, interview_status, notes, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ''', (name, age, major, school, meeting_room, meeting_time,
                      phone, email, position_applied, interview_status, notes))

                person_id = cursor.lastrowid
                conn.commit()

                print(f"Added person: {name} (ID: {person_id})")
                return person_id

        except sqlite3.IntegrityError:
            print(f"Error: Person '{name}' already exists in database")
            raise ValueError(f"Person '{name}' already exists")
        except Exception as e:
            print(f"Error adding person: {e}")
            raise

    def update_person(self, person_id: int = None, name: str = None, **kwargs) -> bool:
        """Update person information with flexible fields"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                if person_id is None and name:
                    cursor.execute('SELECT id FROM people WHERE name = ?', (name,))
                    result = cursor.fetchone()
                    if not result:
                        return False
                    person_id = result[0]

                if person_id is None:
                    return False

                # Build update query dynamically
                update_fields = []
                values = []

                allowed_fields = ['name', 'age', 'major', 'school', 'meeting_room',
                                'meeting_time', 'phone', 'email', 'position_applied',
                                'interview_status', 'notes', 'active']

                for field, value in kwargs.items():
                    if field in allowed_fields and value is not None:
                        update_fields.append(f"{field} = ?")
                        values.append(value)

                if not update_fields:
                    return False

                update_fields.append("updated_at = CURRENT_TIMESTAMP")
                values.append(person_id)

                query = f"UPDATE people SET {', '.join(update_fields)} WHERE id = ?"
                cursor.execute(query, values)

                if cursor.rowcount > 0:
                    conn.commit()
                    return True
                return False

        except Exception as e:
            print(f"Error updating person: {e}")
            return False

    def delete_person(self, person_id: int = None, name: str = None, soft_delete: bool = True) -> bool:
        """Delete person (soft delete by default)"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                if person_id is None and name:
                    cursor.execute('SELECT id FROM people WHERE name = ?', (name,))
                    result = cursor.fetchone()
                    if not result:
                        return False
                    person_id = result[0]

                if person_id is None:
                    return False

                if soft_delete:
                    # Soft delete - set active = FALSE
                    cursor.execute('''
                        UPDATE people 
                        SET active = FALSE, updated_at = CURRENT_TIMESTAMP 
                        WHERE id = ?
                    ''', (person_id,))
                else:
                    # Hard delete - remove from database
                    cursor.execute('DELETE FROM people WHERE id = ?', (person_id,))

                if cursor.rowcount > 0:
                    conn.commit()
                    return True
                return False

        except Exception as e:
            print(f"Error deleting person: {e}")
            return False

    def update_checkin(self, person_id: int) -> bool:
        """Update checkin status + time when candidate is recognized"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                               UPDATE people
                               SET checkin_status = 'checked_in',
                                   checkin_time   = CURRENT_TIMESTAMP,
                                   updated_at     = CURRENT_TIMESTAMP
                               WHERE id = ?
                               ''', (person_id,))
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            print(f"Error updating checkin: {e}")
            return False

    def schedule_interview(self, person_id: int, hr_user_id: int,
                          interview_date: str, interview_room: str,
                          interview_type: str = 'technical',
                          duration_minutes: int = 60,
                          notes: str = None) -> int:
        """Schedule an interview"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute('''
                    INSERT INTO interview_schedules 
                    (person_id, hr_user_id, interview_date, interview_room, 
                     interview_type, duration_minutes, notes)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (person_id, hr_user_id, interview_date, interview_room,
                      interview_type, duration_minutes, notes))

                interview_id = cursor.lastrowid

                # Update person's meeting info
                cursor.execute('''
                    UPDATE people 
                    SET meeting_room = ?, meeting_time = ?, interview_status = 'scheduled'
                    WHERE id = ?
                ''', (interview_room, interview_date, person_id))

                conn.commit()
                return interview_id

        except Exception as e:
            print(f"Error scheduling interview: {e}")
            raise

    def get_interviews(self, date_from: str = None, date_to: str = None,
                      status: str = None) -> List[Dict]:
        """Get interview schedules with filtering"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                query = '''
                    SELECT i.id, i.interview_date, i.interview_room, i.interview_type,
                           i.duration_minutes, i.status, i.notes,
                           p.name as person_name, p.phone, p.email, p.position_applied,
                           h.full_name as hr_name
                    FROM interview_schedules i
                    JOIN people p ON i.person_id = p.id
                    JOIN hr_users h ON i.hr_user_id = h.id
                    WHERE 1=1
                '''

                params = []

                if date_from:
                    query += " AND i.interview_date >= ?"
                    params.append(date_from)

                if date_to:
                    query += " AND i.interview_date <= ?"
                    params.append(date_to)

                if status:
                    query += " AND i.status = ?"
                    params.append(status)

                query += " ORDER BY i.interview_date"

                cursor.execute(query, params)
                results = cursor.fetchall()

                interviews = []
                columns = [description[0] for description in cursor.description]

                for row in results:
                    interview_dict = dict(zip(columns, row))
                    interviews.append(interview_dict)

                return interviews

        except Exception as e:
            print(f"Error getting interviews: {e}")
            return []

    def update_interview_status(self, interview_id: int, status: str,
                               notes: str = None) -> bool:
        """Update interview status"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                if notes:
                    cursor.execute('''
                        UPDATE interview_schedules 
                        SET status = ?, notes = ?, updated_at = CURRENT_TIMESTAMP
                        WHERE id = ?
                    ''', (status, notes, interview_id))
                else:
                    cursor.execute('''
                        UPDATE interview_schedules 
                        SET status = ?, updated_at = CURRENT_TIMESTAMP
                        WHERE id = ?
                    ''', (status, interview_id))

                # Also update person's interview status
                cursor.execute('''
                    UPDATE people p
                    SET interview_status = ?
                    FROM interview_schedules i
                    WHERE p.id = i.person_id AND i.id = ?
                ''', (status, interview_id))

                conn.commit()
                return cursor.rowcount > 0

        except Exception as e:
            print(f"Error updating interview status: {e}")
            return False

    def get_dashboard_stats(self) -> Dict:
        """Get statistics for HR dashboard"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                stats = {}

                # Total candidates
                cursor.execute('SELECT COUNT(*) FROM people WHERE active = TRUE')
                stats['total_candidates'] = cursor.fetchone()[0]

                # Interview stats
                cursor.execute('SELECT status, COUNT(*) FROM interview_schedules GROUP BY status')
                interview_stats = dict(cursor.fetchall())
                stats['interviews'] = interview_stats

                # Today's interviews
                today = datetime.now().strftime('%Y-%m-%d')
                cursor.execute('''
                    SELECT COUNT(*) FROM interview_schedules 
                    WHERE DATE(interview_date) = ? AND status = 'scheduled'
                ''', (today,))
                stats['today_interviews'] = cursor.fetchone()[0]

                # Candidates by status
                cursor.execute('SELECT interview_status, COUNT(*) FROM people GROUP BY interview_status')
                candidate_stats = dict(cursor.fetchalls())
                stats['candidates_by_status'] = candidate_stats

                return stats

        except Exception as e:
            print(f"Error getting dashboard stats: {e}")
            return {}

    # Keep all existing methods from original DatabaseManager
    def get_person(self, person_id: int = None, name: str = None) -> Optional[Dict]:
        """Get person information by ID or name"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                if person_id:
                    cursor.execute('SELECT * FROM people WHERE id = ? AND active = TRUE', (person_id,))
                elif name:
                    cursor.execute('SELECT * FROM people WHERE name = ? AND active = TRUE', (name,))
                else:
                    return None

                result = cursor.fetchone()
                if not result:
                    return None

                columns = [description[0] for description in cursor.description]
                person_dict = dict(zip(columns, result))
                return person_dict

        except Exception as e:
            print(f"Error getting person: {e}")
            return None

    def save_face_embedding(self, person_id: int, embedding: np.ndarray,
                            faces_processed: int = 1) -> int:
        """Save face embedding for a person"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute('SELECT id FROM people WHERE id = ?', (person_id,))
                if not cursor.fetchone():
                    raise ValueError(f"Person ID {person_id} not found")

                embedding_norm = float(np.linalg.norm(embedding))
                embedding_bytes = embedding.tobytes()

                cursor.execute('SELECT id FROM face_embeddings WHERE person_id = ?', (person_id,))
                existing = cursor.fetchone()

                if existing:
                    cursor.execute('''
                        UPDATE face_embeddings 
                        SET embedding = ?, faces_processed = ?, embedding_norm = ?
                        WHERE person_id = ?
                    ''', (embedding_bytes, faces_processed, embedding_norm, person_id))
                    embedding_id = existing[0]
                else:
                    cursor.execute('''
                        INSERT INTO face_embeddings (person_id, embedding, faces_processed, embedding_norm)
                        VALUES (?, ?, ?, ?)
                    ''', (person_id, embedding_bytes, faces_processed, embedding_norm))
                    embedding_id = cursor.lastrowid

                conn.commit()
                return embedding_id

        except Exception as e:
            print(f"Error saving face embedding: {e}")
            raise

    def get_all_embeddings(self):
        """
        Get all face embeddings with person information (with phone column)

        Returns:
            tuple: (embeddings_list, names_list, person_info_list)
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # First, check what columns exist in the people table
                cursor.execute("PRAGMA table_info(people)")
                columns_info = cursor.fetchall()
                available_columns = [col[1] for col in columns_info]
                print(f"DEBUG: Available columns in 'people' table: {available_columns}")

                # Check face_embeddings table structure
                cursor.execute("PRAGMA table_info(face_embeddings)")
                fe_columns_info = cursor.fetchall()
                fe_available_columns = [col[1] for col in fe_columns_info]
                print(f"DEBUG: Available columns in 'face_embeddings' table: {fe_available_columns}")

                # Build query based on available columns
                if 'phone' in available_columns:
                    # Query with phone column
                    query = """
                            SELECT fe.embedding, \
                                   p.id, \
                                   p.name, \
                                   p.age, \
                                   p.major, \
                                   p.school,
                                   p.meeting_room, \
                                   p.meeting_time, \
                                   p.phone, \
                                   fe.faces_processed
                            FROM face_embeddings fe
                                     JOIN people p ON fe.person_id = p.id
                            WHERE p.active = 1
                            ORDER BY p.name \
                            """
                    has_phone_column = True
                else:
                    # Query without phone column
                    query = """
                            SELECT fe.embedding, \
                                   p.id, \
                                   p.name, \
                                   p.age, \
                                   p.major, \
                                   p.school,
                                   p.meeting_room, \
                                   p.meeting_time, \
                                   fe.faces_processed
                            FROM face_embeddings fe
                                     JOIN people p ON fe.person_id = p.id
                            WHERE p.active = 1
                            ORDER BY p.name \
                            """
                    has_phone_column = False
                    print("WARNING: Column 'phone' not found, skipping in query")

                print(f"DEBUG: Executing query...")
                cursor.execute(query)
                results = cursor.fetchall()

                if not results:
                    print("DEBUG: No results returned from query")
                    return [], [], []

                print(f"DEBUG: Query returned {len(results)} results")

                embeddings = []
                names = []
                person_info = []

                for i, row in enumerate(results):
                    try:
                        if has_phone_column:
                            # Extract data from row (with phone)
                            embedding_bytes = row[0]  # fe.embedding
                            person_id = row[1]  # p.id
                            name = row[2]  # p.name
                            age = row[3]  # p.age
                            major = row[4]  # p.major
                            school = row[5]  # p.school
                            meeting_room = row[6]  # p.meeting_room
                            meeting_time = row[7]  # p.meeting_time
                            phone = row[8]  # p.phone
                            faces_processed = row[9]  # fe.faces_processed
                        else:
                            # Extract data from row (without phone)
                            embedding_bytes = row[0]  # fe.embedding
                            person_id = row[1]  # p.id
                            name = row[2]  # p.name
                            age = row[3]  # p.age
                            major = row[4]  # p.major
                            school = row[5]  # p.school
                            meeting_room = row[6]  # p.meeting_room
                            meeting_time = row[7]  # p.meeting_time
                            phone = None  # Default value
                            faces_processed = row[8]  # fe.faces_processed

                        print(f"DEBUG: Processing person {i + 1}: {name}")

                        # Convert embedding bytes to numpy array
                        if embedding_bytes:
                            # ArcFace embeddings are typically 512 float32 values = 2048 bytes
                            embedding_array = np.frombuffer(embedding_bytes, dtype=np.float32)

                            print(f"DEBUG: Embedding array shape: {embedding_array.shape}")

                            # Verify embedding size
                            if len(embedding_array) == 512:  # Expected size for ArcFace
                                embeddings.append(embedding_array)
                                names.append(name)

                                # Build person info dictionary
                                info = {
                                    'id': person_id,
                                    'name': name,
                                    'age': age,
                                    'major': major,
                                    'school': school,
                                    'phone': phone,
                                    'meeting_room': meeting_room,
                                    'meeting_time': meeting_time,
                                    'faces_processed': faces_processed if faces_processed else 1
                                }
                                person_info.append(info)

                                norm = np.linalg.norm(embedding_array)
                                print(
                                    f"DEBUG: ✓ Loaded {name} (phone: {phone}) - embedding shape: {embedding_array.shape}, norm: {norm:.6f}")
                            else:
                                print(
                                    f"WARNING: Invalid embedding size for {name}: {len(embedding_array)}, expected 512")
                        else:
                            print(f"WARNING: No embedding data for {name}")

                    except Exception as row_error:
                        print(f"ERROR processing row {i + 1}: {row_error}")
                        import traceback
                        traceback.print_exc()
                        continue

                print(f"DEBUG: Successfully processed {len(embeddings)} embeddings")
                print(f"DEBUG: Names: {names}")

                return embeddings, names, person_info

        except Exception as e:
            print(f"ERROR in get_all_embeddings: {e}")
            import traceback
            traceback.print_exc()
            return [], [], []


    def list_all_people(self, active_only: bool = True) -> List[Dict]:
        """List all people in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                query = 'SELECT * FROM people'
                if active_only:
                    query += ' WHERE active = TRUE'
                query += ' ORDER BY name'

                cursor.execute(query)
                results = cursor.fetchall()

                columns = [description[0] for description in cursor.description]
                people = []
                for row in results:
                    person_dict = dict(zip(columns, row))
                    people.append(person_dict)

                return people

        except Exception as e:
            print(f"Error listing people: {e}")
            return []

    def get_database_stats(self) -> Dict:
        """Get database statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute('SELECT COUNT(*) FROM people WHERE active = TRUE')
                active_people = cursor.fetchone()[0]

                cursor.execute('SELECT COUNT(*) FROM people')
                total_people = cursor.fetchone()[0]

                cursor.execute('SELECT COUNT(*) FROM face_embeddings')
                total_embeddings = cursor.fetchone()[0]

                cursor.execute('SELECT COUNT(*) FROM hr_users WHERE active = TRUE')
                active_hr_users = cursor.fetchone()[0]

                cursor.execute('SELECT COUNT(*) FROM interview_schedules')
                total_interviews = cursor.fetchone()[0]

                db_size = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0

                return {
                    'active_people': active_people,
                    'total_people': total_people,
                    'total_embeddings': total_embeddings,
                    'active_hr_users': active_hr_users,
                    'total_interviews': total_interviews,
                    'database_size_mb': db_size / (1024 * 1024),
                    'database_path': self.db_path
                }

        except Exception as e:
            print(f"Error getting database stats: {e}")
            return {}