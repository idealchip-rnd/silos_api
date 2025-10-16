import mysql.connector
from datetime import datetime

# --- Database connection ---
conn = mysql.connector.connect(
    host="localhost",
    user="silos_user",
    password="Idealchip123@",
    database="silos"
)
cur = conn.cursor(dictionary=True)

# --- Configuration ---
iteration = 4   # II part of the code
poll_run_id = iteration
sensors_per_cable = 8
target_table = "readings_raw"   # change freely

# --- Step 1: Read silos table ---
cur.execute("SELECT id, silo_number, cable_count FROM silos ORDER BY silo_number")
silos = cur.fetchall()

# --- Step 2: Calculate totals ---
total_silos = len(silos)
total_cables = sum(s["cable_count"] or 0 for s in silos)
total_readings = total_cables * sensors_per_cable

print("\n=== Injection Preview ===")
print(f"Destination table: {target_table}")
print(f"Iteration number:  {iteration}")
print(f"Silos found:       {total_silos}")
print(f"Total cables:      {total_cables}")
print(f"Sensors per cable: {sensors_per_cable}")
print(f"Total readings:    {total_readings:,}\n")

if total_readings == 0:
    print("No cables detected. Aborting.")
    cur.close()
    conn.close()
    exit()

avg_cables_per_silo = total_cables / total_silos
print(f"Average cables per silo: {avg_cables_per_silo:.2f}")
print("Example reading code: IIXXXY.Z (e.g. 0112301.5 → iteration 01, silo 123, cable 0, sensor 5)\n")

# --- Step 3: Confirmation ---
confirm = input(f"Proceed with injecting into '{target_table}'? (y/n): ").strip().lower()
if confirm != "y":
    print("❌ Operation cancelled.")
    cur.close()
    conn.close()
    exit()

# --- Step 4: Inject readings ---
print("\nInjecting... please wait.")
insert_count = 0
missing_sensors = 0

for silo in silos:
    silo_no = silo["silo_number"]
    cable_count = silo["cable_count"] or 0

    for cable_index in range(cable_count):
        for sensor_index in range(sensors_per_cable):
            code = f"{iteration:02d}{silo_no:03d}{cable_index}.{sensor_index}"
            value_c = float(code)
            polled_at = datetime.now()

            # Proper join through cables
            cur.execute("""
                SELECT s.id AS sensor_id
                FROM sensors s
                JOIN cables c ON s.cable_id = c.id
                JOIN silos si ON c.silo_id = si.id
                WHERE si.silo_number = %s
                  AND c.cable_index = %s
                  AND s.sensor_index = %s
                LIMIT 1
            """, (silo_no, cable_index, sensor_index))
            row = cur.fetchone()

            if row:
                query = f"""
                    INSERT INTO {target_table} (sensor_id, value_c, polled_at, poll_run_id)
                    VALUES (%s, %s, %s, %s)
                """
                cur.execute(query, (row["sensor_id"], value_c, polled_at, poll_run_id))
                insert_count += 1
            else:
                missing_sensors += 1
                print(f"⚠️ No sensor found for Silo {silo_no}, Cable {cable_index}, Sensor {sensor_index}")

conn.commit()
cur.close()
conn.close()

print(f"\n✅ Injection complete: {insert_count:,} readings inserted.")
if missing_sensors:
    print(f"⚠️ {missing_sensors} sensors were not found in database.")
