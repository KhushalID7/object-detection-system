import psycopg2

import time


# Connect to DB
conn = psycopg2.connect(dbname="DEMOimg", user="postgres", password="1234")
cur = conn.cursor()

# Read image
with open("img.jpg", "rb") as f:
    image_bytes = f.read()

# Insert into DB
try:
    while True:
        cur.execute(
            "INSERT INTO add_demo_img (name, image_data) VALUES (%s, %s)",
            ("Khushal demo pic", psycopg2.Binary(image_bytes))
        )
        conn.commit()
        print("Image inserted.")
        time.sleep(2)  # Optional: Pause for 2 seconds before next insert

except KeyboardInterrupt:
    print("Stopped by user.")


cur.close()
conn.close()