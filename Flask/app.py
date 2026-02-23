# ui liberary
# main server file
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy # type: ignore
from werkzeug.security import generate_password_hash

app = Flask(__name__)

app.secret_key = 'secretkey'#secret key for session management
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEM_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app) # type: ignore #initialie the database

# schema 
class User(db.Model):
      id = db.Column(db.Integer, primary_key = True)
      username = db.Column(db.String(150), unique = True, nullable = False)
      email = db.Column(db.String(150), unique = True, nullable =False)
      password = db.Column(db.String(150), nullable = False)

with app.app_context():
      db.create_all() #create the database tables


@app.route("/")
def home():
      return render_template("index.html")

@app.route("/login")
def login():
      return render_template("login.html")

@app.route("/register", methods = ['GET' , 'POST'])
def register():
      if request.method == 'POST':
            username = request.form['username']    
            email = request.form['email']   
            password = request.form['password']   
            confirm_password = request.form['confirm_password']

            # create a new user
            hashed_password = generate_password_hash(password)
            new_user = User(
                  username = username.strip(),
                  email = email.strip(),
                  password = hashed_password
            )     
            try:
                  db.session.add(new_user)
                  db.session.commit()
                  flash('Registration successful ! please log in.' , 'sucess')
                  return redirect(url_for(login))
            except Exception as e :
                  db.session.rollback()
                  flash("An error occured during registration. please try again." ,"danger" )
                  return redirect(url_for('register'))
      return render_template("register.html")




# these lines should be in end
if __name__ == '__main__':
      app.run(debug=True)