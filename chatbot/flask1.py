"""#protocol domain path part query parameters
#root path
#entry point or home page of application

from flask import Flask,jsonify
import requests
app = Flask(__name__)

@app.route("/he")
def hel():
    return "hello"

@app.route('/users')
def user():
    return "bye"  

@app.route('/')
def home():
    return 'no '

@app.route('/users/id',methods=['POST'])
def addUser(name):
    data = requests.get_json()
    name = data.get('name')
    return jsonify(name)
app.run()"""


from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route("/he")
def hel():
    return "hello"



@app.route('/')
def home():
    return 'no '

@app.route('/users/id', methods=['POST'])
def addUser():
    # get JSON data sent by the client
    data = request.get_json()  # ✅ from flask.request

    # extract 'name' from that JSON
    name = data.get('name', 'NoName')

    # send back a JSON response
    return jsonify({"message": f"User {name} added successfully!"})

users = []
@app.route('/users', methods=['GET', 'POST'])
def manage_users():
    if request.method == 'POST':
        # Read JSON body sent by client
        data = request.get_json()
        name = data.get('name', 'NoName')

        # Add user to list
        users.append(name)
        return jsonify({"message": f"User {name} added successfully!"}), 201

    elif request.method == 'GET':
        # Return all users as JSON
        return jsonify({"users": users})

if __name__ == '__main__':
    app.run(debug=True)
