import firebase_admin
from firebase_admin import credentials
from firebase_admin import db


cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred,{
    'databaseURL':"https://faceattendence-554d6-default-rtdb.firebaseio.com/"
})
ref=db.reference('Students')

data={
    "651269":
    {
        "name": "Rahul Mondal",
        "major": "CSE",
        "starting_year": 2022,
        "total_attendance": 7,
        "standing": "B",
        "year": 4,
        "last_attendance_time": "2022-12-11 00:54:34"
    },
    "852741":
        {
            "name": "Emly Blunt",
            "major": "Economics",
            "starting_year": 2020,
            "total_attendance": 12,
            "standing": "A",
            "year": 5,
            "last_attendance_time": "2022-12-11 00:54:34"
        },
        "963852":
        {
            "name": "Elon Musk",
            "major": "Physics",
            "starting_year": 2020,
            "total_attendance": 7,
            "standing": "C",
            "year": 2,
            "last_attendance_time": "2022-12-11 00:54:34"
        }
}
for key,value in data.items():
    ref.child(key).set(value)