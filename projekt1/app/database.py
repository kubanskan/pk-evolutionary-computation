import sqlite3
from datetime import datetime
from typing import List, Dict


class DataBase:

    def __init__(self, db_file: str = "ga_results.db"):
        self.db_file = db_file
        self._create_tables()

    def _create_tables(self):
        """Stwórz tabele"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                function_name TEXT NOT NULL,
                num_variables INTEGER,
                best_fitness REAL,
                elapsed_time REAL
            )
        ''')


        cursor.execute('''
            CREATE TABLE IF NOT EXISTS iterations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER,
                generation INTEGER,
                best_fitness REAL,
                avg_fitness REAL,
                std_fitness REAL,
                FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE CASCADE
            )
        ''')

        conn.commit()
        conn.close()

    def save_run(self, function_name: str, num_variables: int, result: dict) -> int:
        """
        Zapisz uruchomienie i jego iteracje.
        """
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        cursor.execute('''
            INSERT INTO runs (timestamp, function_name, num_variables, best_fitness, elapsed_time)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            timestamp,
            function_name,
            num_variables,
            result['best_fitness'],
            result['elapsed_time']
        ))

        run_id = cursor.lastrowid

        history = result['history']
        for i in range(len(history['generation'])):
            cursor.execute('''
                INSERT INTO iterations (run_id, generation, best_fitness, avg_fitness, std_fitness)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                run_id,
                history['generation'][i],
                history['best_fitness'][i],
                history['avg_fitness'][i],
                history['std_fitness'][i]
            ))

        conn.commit()
        conn.close()

        print(f"✓ Zapisano uruchomienie ID={run_id} z {len(history['generation'])} iteracjami do bazy")
        return run_id

    def get_all_runs(self) -> List[Dict]:
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT id, timestamp, function_name, num_variables, best_fitness, elapsed_time
            FROM runs
            ORDER BY timestamp DESC
        ''')

        runs = []
        for row in cursor.fetchall():
            runs.append({
                'id': row[0],
                'timestamp': row[1],
                'function_name': row[2],
                'num_variables': row[3],
                'best_fitness': row[4],
                'elapsed_time': row[5]
            })

        conn.close()
        return runs

    def get_iterations(self, run_id: int) -> List[Dict]:
        """Pobierz wszystkie iteracje dla danego uruchomienia"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT generation, best_fitness, avg_fitness, std_fitness
            FROM iterations
            WHERE run_id = ?
            ORDER BY generation
        ''', (run_id,))

        iterations = []
        for row in cursor.fetchall():
            iterations.append({
                'generation': row[0],
                'best_fitness': row[1],
                'avg_fitness': row[2],
                'std_fitness': row[3]
            })

        conn.close()
        return iterations

    def export_iterations_to_csv(self, run_id: int, filename: str):
        """Eksportuj iteracje do CSV"""
        import csv

        iterations = self.get_iterations(run_id)

        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Generation', 'Best_Fitness', 'Avg_Fitness', 'Std_Fitness'])

            for it in iterations:
                writer.writerow([
                    it['generation'],
                    it['best_fitness'],
                    it['avg_fitness'],
                    it['std_fitness']
                ])

        print(f"✓ Wyeksportowano {len(iterations)} iteracji do {filename}")

    def delete_run(self, run_id: int):
        """Usuń uruchomienie (wraz z iteracjami)"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM runs WHERE id = ?', (run_id,))
        conn.commit()
        conn.close()

    def clear_all(self):
        """Wyczyść całą bazę"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM iterations')
        cursor.execute('DELETE FROM runs')
        conn.commit()
        conn.close()