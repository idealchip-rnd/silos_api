from flask import Flask, jsonify
from flask_cors import CORS    # ← import this
import mysql.connector
from mysql.connector import Error

app = Flask(__name__)
CORS(app)   # ← enable CORS for all routes


def fetch_latest_per_sensor():
    """Fetch the latest reading for each sensor."""
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="silos_user",
            password="Idealchip123@",
            database="silos"
        )

        if conn.is_connected():
            cursor = conn.cursor(dictionary=True)
            query = """
            SELECT r.sensor_id, r.value_c, r.polled_at
            FROM readings_raw r
            JOIN (
                SELECT sensor_id, MAX(polled_at) AS latest
                FROM readings_raw
                GROUP BY sensor_id
            ) x ON r.sensor_id = x.sensor_id AND r.polled_at = x.latest
            ORDER BY r.sensor_id ASC;
            """
            cursor.execute(query)
            rows = cursor.fetchall()
            print(f"Fetched {len(rows)} latest sensor readings.")
            return rows

    except Error as e:
        print("Database error:", e)
        return []

    finally:
        if 'conn' in locals() and conn.is_connected():
            cursor.close()
            conn.close()

@app.route("/")
def home():
    return "<h2>Silo Temperature API is running 🚀</h2><p>Use <code>/api/all_sensors</code> to get latest readings.</p>"

@app.route("/api/all_sensors")
def all_sensors():
    """Return JSON array of latest readings in order by sensor_id."""
    data = fetch_latest_per_sensor()

    # You can either return full data...
    # return jsonify(data)

    # ...or only temperature values if frontend expects an array of 1800 values
    readings = [row["value_c"] for row in data]
    return jsonify(readings)

if __name__ == "__main__":
    # Accessible from local network too
    app.run(host="0.0.0.0", port=5000, debug=True)
