# השתמש בתמונת Python רשמית
FROM python:3.11-slim

# הגדר תיקיית עבודה בקונטיינר
WORKDIR /app

# העתק את קובץ הדרישות
COPY requirements.txt .

# התקן את החבילות הנדרשות
RUN pip install --no-cache-dir -r requirements.txt

# העתק את תיקיית השרת ואת קבצי הנתונים
COPY server/ ./server/
COPY data/ ./data/

# חשוף את הפורט שבו השרת ירוץ
EXPOSE 8000

# הפקודה להתחלת השרת - נרוץ מתיקיית השורש כדי שהנתיבים לקבצי הנתונים יעבדו
CMD ["uvicorn", "server.app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
