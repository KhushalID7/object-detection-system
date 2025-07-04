import psycopg2
import numpy as np
import cv2

# Connect to DB
conn = psycopg2.connect(dbname="DEMOimg", user="postgres", password="1234")
cur = conn.cursor()

cur.execute("SELECT image_data FROM detection_record WHERE id = %s", (500,))
image_data = cur.fetchone()[0]  # Adjust index based on your table structure
# Convert binary to numpy array
nparr = np.frombuffer(image_data, np.uint8)

# Decode image
img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

#img = cv2.resize(img, (300, 400))

# Display image
cv2.imshow('Image from DB', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Cleanup
cur.close()
conn.close()