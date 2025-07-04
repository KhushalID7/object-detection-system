import psycopg2




# Connect to DB
conn = psycopg2.connect(dbname="DEMOimg", user="postgres", password="1234")
cur = conn.cursor()

# Read image
with open("img.jpg", "rb") as f:
    image_bytes = f.read()

# Insert into DB
cur.execute("INSERT INTO add_demo_img (name, image_data) VALUES (%s, %s)", ("Khushal demo pic", psycopg2.Binary(image_bytes)))
conn.commit()




cur.close()
conn.close()
