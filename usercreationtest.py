from app import app
from models import db, User

with app.app_context():
    # Create new user object
    admin = User(username='bashar')
    admin.set_password('bashar')
    admin.set_role('operator')

    # Add to the database
    db.session.add(admin)
    db.session.commit()

    print("✅ User created successfully.")
