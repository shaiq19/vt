from flask import Flask, flash, request, redirect, url_for, render_template
import urllib.request
import os
from werkzeug.utils import secure_filename
import sqlite3

import numpy as np
import cv2  # Library for image processing
app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads/'
usermodel=''
pic=''

app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def home():
    return render_template('index.html')



@app.route('/pink',methods=['POST'] )
def pink():
    usermodel = 'pink3.jpg'
    global conn, cursor
    conn = sqlite3.connect("db_FINAL.db")
    cursor = conn.cursor()
    cursor.execute(
        "CREATE TABLE IF NOT EXISTS `databs` (mem_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL, model TEXT,low1 INT,low2 INT,low3 INT,High1 INT,High2 INT,High3 INT)")
    # sql_update_query = """insert into databs (model,low1,High1) values('arsalan','1','1')"""

    # cursor.execute(sql_update_query)
    print("added")
    sql_update_query = """Update databs set model = ?,low1=?,low2=?,low3=?,High1=?,High2=?,High3=? where mem_id = ?"""
    data = (usermodel,42,108,90,255, 255, 255, 1)

    cursor.execute(sql_update_query, data)

    conn.commit()

    return render_template('index.html')

@app.route('/red',methods=['POST'] )
def red():
    usermodel = 'reeed.png'
    global conn, cursor
    conn = sqlite3.connect("db_FINAL.db")
    cursor = conn.cursor()
    cursor.execute(
        "CREATE TABLE IF NOT EXISTS `databs` (mem_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL, model TEXT,low1 INT,low2 INT,low3 INT,High1 INT,High2 INT,High3 INT)")
    # sql_update_query = """insert into databs (model,low1,High1) values('arsalan','1','1')"""

    # cursor.execute(sql_update_query)
    print("added")
    sql_update_query = """Update databs set model = ?,low1=?,low2=?,low3=?,High1=?,High2=?,High3=? where mem_id = ?"""
    data = (usermodel, 42, 108, 90, 255, 255, 255, 1)

    cursor.execute(sql_update_query, data)

    conn.commit()

    return render_template('index.html')


@app.route('/yellow',methods=['POST'] )
def yellow():
    usermodel='yellow2.jpg'
    global conn, cursor
    conn = sqlite3.connect("db_FINAL.db")
    cursor = conn.cursor()
    cursor.execute(
        "CREATE TABLE IF NOT EXISTS `databs` (mem_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL, model TEXT,low1 INT,low2 INT,low3 INT,High1 INT,High2 INT,High3 INT)")
    #sql_update_query = """insert into databs (model,low1,High1) values('arsalan','1','1')"""

    #cursor.execute(sql_update_query)
    print("added")

    sql_update_query = """Update databs set model = ?,low1=?,low2=?,low3=?,High1=?,High2=?,High3=? where mem_id = ?"""
    data = (usermodel, 16, 54, 3, 139 ,255 ,255, 1)

    cursor.execute(sql_update_query, data)

    conn.commit()

    return render_template('index.html')

@app.route('/green',methods=['POST'] )
def green():
    usermodel='green2.jpg'
    global conn, cursor
    conn = sqlite3.connect("db_FINAL.db")
    cursor = conn.cursor()
    cursor.execute(
        "CREATE TABLE IF NOT EXISTS `databs` (mem_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL, model TEXT,low1 INT,low2 INT,low3 INT,High1 INT,High2 INT,High3 INT)")
    #sql_update_query = """insert into databs (model,low1,High1) values('arsalan','1','1')"""

    #cursor.execute(sql_update_query)
    print("added")

    sql_update_query = """Update databs set model = ?,low1=?,low2=?,low3=?,High1=?,High2=?,High3=? where mem_id = ?"""
    data = (usermodel, 25, 52, 70, 102, 255, 255, 1)

    cursor.execute(sql_update_query, data)

    conn.commit()

    return render_template('index.html')



@app.route('/', methods=['POST'])
def upload_image():





    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        global conn, cursor
        conn = sqlite3.connect("db_FINAL.db")
        cursor = conn.cursor()
        cursor.execute(
            "CREATE TABLE IF NOT EXISTS `databs` (mem_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL, model TEXT,low1 INT,low2 INT,low3 INT,High1 INT,High2 INT,High3 INT)")
        # sql_update_query = """insert into databs (model,low1,High1) values('arsalan','1','1')"""

        # cursor.execute(sql_update_query)
        print("added")
        sql_update_query = """Update databs set model = ?,low1=?,low2=?,low3=?,High1=?,High2=?,High3=? where mem_id = ?"""
        data = (filename, 42, 108, 90, 255, 255, 255, 1)

        cursor.execute(sql_update_query, data)

        conn.commit()

    

        # print('upload_image filename: ' + filename)
        flash('Image successfully uploaded and displayed below')
        frame = cv2.imread('static/uploads/'+filename+'')
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # define range of green color in HSV
        # lower_green = np.array([25, 52, 72])
        # upper_green = np.array([102, 255, 255])
        # lower_blue = np.array([110,50,50])
        # upper_blue = np.array([130,255,255])
        low_yell = np.array([16, 54, 3])
        high_yell = np.array([139, 255, 255])
        low_pin = np.array([42, 108, 90])
        high_pink = np.array([255, 255, 255])
        low_black = np.array([0, 0, 0])
        high_black = np.array([187, 255, 38])

        low_red = np.array([161, 155, 84])
        high_red = np.array([179, 255, 255])
        mask_white = cv2.inRange(hsv, low_red, high_red)
        mask_black = cv2.bitwise_not(mask_white)

        # converting mask_black to 3 channels
        W, L = mask_black.shape
        mask_black_3CH = np.empty((W, L, 3), dtype=np.uint8)
        mask_black_3CH[:, :, 0] = mask_black
        mask_black_3CH[:, :, 1] = mask_black
        mask_black_3CH[:, :, 2] = mask_black

        #cv2.imshow('orignal Image', frame)
        # cv2.imshow('mask_black',mask_black_3CH)

        dst3 = cv2.bitwise_and(mask_black_3CH, frame)
        # cv2.imshow('Pic+mask_inverse',dst3)

        # ///////
        W, L = mask_white.shape
        mask_white_3CH = np.empty((W, L, 3), dtype=np.uint8)
        mask_white_3CH[:, :, 0] = mask_white
        mask_white_3CH[:, :, 1] = mask_white
        mask_white_3CH[:, :, 2] = mask_white

        # cv2.imshow('Wh_mask',mask_white_3CH)
        dst3_wh = cv2.bitwise_or(mask_white_3CH, dst3)
        # cv2.imshow('Pic+mask_wh',dst3_wh)

        # /////////////////
        # imag 5,15,17,16
        # changing for design
        design = cv2.imread('static/uploads/6_2_1.png')
        design = cv2.resize(design, mask_black.shape[1::-1])
        # cv2.imshow('Shirt Design', design)

        design_mask_mixed = cv2.bitwise_or(mask_black_3CH, design)
        # cv2.imshow('design_mask_mixed',design_mask_mixed)

        final_mask_black_3CH = cv2.bitwise_and(design_mask_mixed, dst3_wh)

        outpath = "Output.jpg"
        # save the image
        cv2.imwrite('static/'+outpath, final_mask_black_3CH)
        filename=secure_filename(outpath)
       # cv2.imshow('Final Output',final_mask_black_3CH)


       # cv2.waitKey()

       # cv2.cv2.destroyAllWindows()  # Destroys all the windows created by imshow

        return render_template('index.html', filename=filename)
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)


@app.route('/pic1/' ,methods=['POST'])
def pic1():
    sqliteConnection = sqlite3.connect('db_FINAL.db')
    cursor = sqliteConnection.cursor()
    print("Connected to SQLite")
    sqlite_select_query = """SELECT * from databs where mem_id=1"""
    cursor.execute(sqlite_select_query)
    records = cursor.fetchall()
    for row in records:
        usermodel=row[1]
        low1 =row[2]
        low2 =row[3]
        low3=row[4]
        h1=row[5]
        h2=row[6]
        h3=row[7]


    cursor.close()
    flash('Image successfully uploaded and displayed below')
    frame = cv2.imread('static/uploads/'+usermodel)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of green color in HSV
    # lower_green = np.array([25, 52, 72])
    # upper_green = np.array([102, 255, 255])
    # lower_blue = np.array([110,50,50])
    # upper_blue = np.array([130,255,255])
    low_yell = np.array([low1, low2, low3])
    high_yell = np.array([h1, h2, h3])
    low_pin = np.array([42, 108, 90])
    high_pink = np.array([255, 255, 255])
   # low_black = np.array([0, 0, 0])
    # high_black = np.array([187, 255, 38])
    print(low_yell)
    print(high_yell)

   # print(low_mode)
   # print(high_pink)
    mask_white = cv2.inRange(hsv, low_yell, high_yell)
    mask_black = cv2.bitwise_not(mask_white)

    # converting mask_black to 3 channels
    W, L = mask_black.shape
    mask_black_3CH = np.empty((W, L, 3), dtype=np.uint8)
    mask_black_3CH[:, :, 0] = mask_black
    mask_black_3CH[:, :, 1] = mask_black
    mask_black_3CH[:, :, 2] = mask_black

    # cv2.imshow('orignal Image', frame)
    # cv2.imshow('mask_black',mask_black_3CH)

    dst3 = cv2.bitwise_and(mask_black_3CH, frame)
    # cv2.imshow('Pic+mask_inverse',dst3)

    # ///////
    W, L = mask_white.shape
    mask_white_3CH = np.empty((W, L, 3), dtype=np.uint8)
    mask_white_3CH[:, :, 0] = mask_white
    mask_white_3CH[:, :, 1] = mask_white
    mask_white_3CH[:, :, 2] = mask_white

    # cv2.imshow('Wh_mask',mask_white_3CH)
    dst3_wh = cv2.bitwise_or(mask_white_3CH, dst3)
    # cv2.imshow('Pic+mask_wh',dst3_wh)

    # /////////////////
    # imag 5,15,17,16
    # changing for design
    design = cv2.imread('static/uploads/6_1_5.png')
    design = cv2.resize(design, mask_black.shape[1::-1])
    # cv2.imshow('Shirt Design', design)

    design_mask_mixed = cv2.bitwise_or(mask_black_3CH, design)
    # cv2.imshow('design_mask_mixed',design_mask_mixed)

    final_mask_black_3CH = cv2.bitwise_and(design_mask_mixed, dst3_wh)

    outpath = "Output.jpg"
    # save the image
    cv2.imwrite('static/' + outpath, final_mask_black_3CH)
    filename = secure_filename(outpath)
    # cv2.imshow('Final Output',final_mask_black_3CH)

    # cv2.waitKey()

    # cv2.cv2.destroyAllWindows()  # Destroys all the windows created by imshow

    return render_template('index.html', filename=filename)

@app.route('/pic2/' ,methods=['POST'])
def pic2():
    sqliteConnection = sqlite3.connect('db_FINAL.db')
    cursor = sqliteConnection.cursor()
    print("Connected to SQLite")
    sqlite_select_query = """SELECT * from databs where mem_id=1"""
    cursor.execute(sqlite_select_query)
    records = cursor.fetchall()
    for row in records:
        usermodel=row[1]
        low1 = row[2]
        low2 = row[3]
        low3=row[4]
        h1=row[5]
        h2=row[6]
        h3=row[7]


    cursor.close()
    flash('Image successfully uploaded and displayed below')
    frame = cv2.imread('static/uploads/'+usermodel)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of green color in HSV
    # lower_green = np.array([25, 52, 72])
    # upper_green = np.array([102, 255, 255])
    # lower_blue = np.array([110,50,50])
    # upper_blue = np.array([130,255,255])
    low_yell = np.array([low1, low2, low3])
    high_yell = np.array([h1, h2, h3])
    low_pin = np.array([42, 108, 90])
    high_pink = np.array([255, 255, 255])
    mask_white = cv2.inRange(hsv, low_yell, high_yell)
    mask_black = cv2.bitwise_not(mask_white)

    # converting mask_black to 3 channels
    W, L = mask_black.shape
    mask_black_3CH = np.empty((W, L, 3), dtype=np.uint8)
    mask_black_3CH[:, :, 0] = mask_black
    mask_black_3CH[:, :, 1] = mask_black
    mask_black_3CH[:, :, 2] = mask_black

    # cv2.imshow('orignal Image', frame)
    # cv2.imshow('mask_black',mask_black_3CH)

    dst3 = cv2.bitwise_and(mask_black_3CH, frame)
    # cv2.imshow('Pic+mask_inverse',dst3)

    # ///////
    W, L = mask_white.shape
    mask_white_3CH = np.empty((W, L, 3), dtype=np.uint8)
    mask_white_3CH[:, :, 0] = mask_white
    mask_white_3CH[:, :, 1] = mask_white
    mask_white_3CH[:, :, 2] = mask_white
    dst3_wh = cv2.bitwise_or(mask_white_3CH, dst3)
    design = cv2.imread('static/uploads/6_1_4.png')
    design = cv2.resize(design, mask_black.shape[1::-1])
    # cv2.imshow('Shirt Design', design)

    design_mask_mixed = cv2.bitwise_or(mask_black_3CH, design)
    # cv2.imshow('design_mask_mixed',design_mask_mixed)

    final_mask_black_3CH = cv2.bitwise_and(design_mask_mixed, dst3_wh)

    outpath = "Output.jpg"
    # save the image
    cv2.imwrite('static/' + outpath, final_mask_black_3CH)
    filename = secure_filename(outpath)
    # cv2.imshow('Final Output',final_mask_black_3CH)

    # cv2.waitKey()

    # cv2.cv2.destroyAllWindows()  # Destroys all the windows created by imshow

    return render_template('index.html', filename=filename)

@app.route('/pic3/' ,methods=['POST'])
def pic3():
    sqliteConnection = sqlite3.connect('db_FINAL.db')
    cursor = sqliteConnection.cursor()
    print("Connected to SQLite")
    sqlite_select_query = """SELECT * from databs where mem_id=1"""
    cursor.execute(sqlite_select_query)
    records = cursor.fetchall()
    for row in records:
        usermodel=row[1]
        low1 = row[2]
        low2 = row[3]
        low3=row[4]
        h1=row[5]
        h2=row[6]
        h3=row[7]


    cursor.close()
    flash('Image successfully uploaded and displayed below')
    frame = cv2.imread('static/uploads/'+usermodel)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of green color in HSV
    # lower_green = np.array([25, 52, 72])
    # upper_green = np.array([102, 255, 255])
    # lower_blue = np.array([110,50,50])
    # upper_blue = np.array([130,255,255])
    low_yell = np.array([low1, low2, low3])
    high_yell = np.array([h1, h2, h3])
    low_pin = np.array([42, 108, 90])
    high_pink = np.array([255, 255, 255])
    mask_white = cv2.inRange(hsv, low_yell, high_yell)
    mask_black = cv2.bitwise_not(mask_white)

    # converting mask_black to 3 channels
    W, L = mask_black.shape
    mask_black_3CH = np.empty((W, L, 3), dtype=np.uint8)
    mask_black_3CH[:, :, 0] = mask_black
    mask_black_3CH[:, :, 1] = mask_black
    mask_black_3CH[:, :, 2] = mask_black

    # cv2.imshow('orignal Image', frame)
    # cv2.imshow('mask_black',mask_black_3CH)

    dst3 = cv2.bitwise_and(mask_black_3CH, frame)
    # cv2.imshow('Pic+mask_inverse',dst3)

    # ///////
    W, L = mask_white.shape
    mask_white_3CH = np.empty((W, L, 3), dtype=np.uint8)
    mask_white_3CH[:, :, 0] = mask_white
    mask_white_3CH[:, :, 1] = mask_white
    mask_white_3CH[:, :, 2] = mask_white
    dst3_wh = cv2.bitwise_or(mask_white_3CH, dst3)
    design = cv2.imread('static/uploads/6_2_1.png')
    design = cv2.resize(design, mask_black.shape[1::-1])
    # cv2.imshow('Shirt Design', design)

    design_mask_mixed = cv2.bitwise_or(mask_black_3CH, design)
    # cv2.imshow('design_mask_mixed',design_mask_mixed)

    final_mask_black_3CH = cv2.bitwise_and(design_mask_mixed, dst3_wh)

    outpath = "Output.jpg"
    # save the image
    cv2.imwrite('static/' + outpath, final_mask_black_3CH)
    filename = secure_filename(outpath)
    # cv2.imshow('Final Output',final_mask_black_3CH)

    # cv2.waitKey()

    # cv2.cv2.destroyAllWindows()  # Destroys all the windows created by imshow

    return render_template('index.html', filename=filename)

@app.route('/pic4/' ,methods=['POST'])
def pic4():
    sqliteConnection = sqlite3.connect('db_FINAL.db')
    cursor = sqliteConnection.cursor()
    print("Connected to SQLite")
    sqlite_select_query = """SELECT * from databs where mem_id=1"""
    cursor.execute(sqlite_select_query)
    records = cursor.fetchall()
    for row in records:
        usermodel=row[1]
        low1 = row[2]
        low2 = row[3]
        low3=row[4]
        h1=row[5]
        h2=row[6]
        h3=row[7]


    cursor.close()
    flash('Image successfully uploaded and displayed below')
    frame = cv2.imread('static/uploads/'+usermodel)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of green color in HSV
    # lower_green = np.array([25, 52, 72])
    # upper_green = np.array([102, 255, 255])
    # lower_blue = np.array([110,50,50])
    # upper_blue = np.array([130,255,255])
    low_yell = np.array([low1, low2, low3])
    high_yell = np.array([h1, h2, h3])
    low_pin = np.array([42, 108, 90])
    high_pink = np.array([255, 255, 255])
    mask_white = cv2.inRange(hsv, low_yell, high_yell)
    mask_black = cv2.bitwise_not(mask_white)

    # converting mask_black to 3 channels
    W, L = mask_black.shape
    mask_black_3CH = np.empty((W, L, 3), dtype=np.uint8)
    mask_black_3CH[:, :, 0] = mask_black
    mask_black_3CH[:, :, 1] = mask_black
    mask_black_3CH[:, :, 2] = mask_black

    # cv2.imshow('orignal Image', frame)
    # cv2.imshow('mask_black',mask_black_3CH)

    dst3 = cv2.bitwise_and(mask_black_3CH, frame)
    # cv2.imshow('Pic+mask_inverse',dst3)

    # ///////
    W, L = mask_white.shape
    mask_white_3CH = np.empty((W, L, 3), dtype=np.uint8)
    mask_white_3CH[:, :, 0] = mask_white
    mask_white_3CH[:, :, 1] = mask_white
    mask_white_3CH[:, :, 2] = mask_white
    dst3_wh = cv2.bitwise_or(mask_white_3CH, dst3)
    design = cv2.imread('static/uploads/001048_1.jpg')
    design = cv2.resize(design, mask_black.shape[1::-1])
    # cv2.imshow('Shirt Design', design)

    design_mask_mixed = cv2.bitwise_or(mask_black_3CH, design)
    # cv2.imshow('design_mask_mixed',design_mask_mixed)

    final_mask_black_3CH = cv2.bitwise_and(design_mask_mixed, dst3_wh)

    outpath = "Output.jpg"
    # save the image
    cv2.imwrite('static/' + outpath, final_mask_black_3CH)
    filename = secure_filename(outpath)
    # cv2.imshow('Final Output',final_mask_black_3CH)

    # cv2.waitKey()

    # cv2.cv2.destroyAllWindows()  # Destroys all the windows created by imshow

    return render_template('index.html', filename=filename)
@app.route('/pic5/' ,methods=['POST'])
def pic5():
    sqliteConnection = sqlite3.connect('db_FINAL.db')
    cursor = sqliteConnection.cursor()
    print("Connected to SQLite")
    sqlite_select_query = """SELECT * from databs where mem_id=1"""
    cursor.execute(sqlite_select_query)
    records = cursor.fetchall()
    for row in records:
        usermodel=row[1]
        low1 = row[2]
        low2 = row[3]
        low3=row[4]
        h1=row[5]
        h2=row[6]
        h3=row[7]


    cursor.close()
    flash('Image successfully uploaded and displayed below')
    frame = cv2.imread('static/uploads/'+usermodel)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of green color in HSV
    # lower_green = np.array([25, 52, 72])
    # upper_green = np.array([102, 255, 255])
    # lower_blue = np.array([110,50,50])
    # upper_blue = np.array([130,255,255])
    low_yell = np.array([low1, low2, low3])
    high_yell = np.array([h1, h2, h3])
    low_pin = np.array([42, 108, 90])
    high_pink = np.array([255, 255, 255])
    mask_white = cv2.inRange(hsv, low_yell, high_yell)
    mask_black = cv2.bitwise_not(mask_white)

    # converting mask_black to 3 channels
    W, L = mask_black.shape
    mask_black_3CH = np.empty((W, L, 3), dtype=np.uint8)
    mask_black_3CH[:, :, 0] = mask_black
    mask_black_3CH[:, :, 1] = mask_black
    mask_black_3CH[:, :, 2] = mask_black

    # cv2.imshow('orignal Image', frame)
    # cv2.imshow('mask_black',mask_black_3CH)

    dst3 = cv2.bitwise_and(mask_black_3CH, frame)
    # cv2.imshow('Pic+mask_inverse',dst3)

    # ///////
    W, L = mask_white.shape
    mask_white_3CH = np.empty((W, L, 3), dtype=np.uint8)
    mask_white_3CH[:, :, 0] = mask_white
    mask_white_3CH[:, :, 1] = mask_white
    mask_white_3CH[:, :, 2] = mask_white
    dst3_wh = cv2.bitwise_or(mask_white_3CH, dst3)
    design = cv2.imread('static/uploads/001101_1.jpg')
    design = cv2.resize(design, mask_black.shape[1::-1])
    # cv2.imshow('Shirt Design', design)

    design_mask_mixed = cv2.bitwise_or(mask_black_3CH, design)
    # cv2.imshow('design_mask_mixed',design_mask_mixed)

    final_mask_black_3CH = cv2.bitwise_and(design_mask_mixed, dst3_wh)

    outpath = "Output.jpg"
    # save the image
    cv2.imwrite('static/' + outpath, final_mask_black_3CH)
    filename = secure_filename(outpath)
    # cv2.imshow('Final Output',final_mask_black_3CH)

    # cv2.waitKey()

    # cv2.cv2.destroyAllWindows()  # Destroys all the windows created by imshow

    return render_template('index.html', filename=filename)

@app.route('/pic7/' ,methods=['POST'])
def pic7():
    sqliteConnection = sqlite3.connect('db_FINAL.db')
    cursor = sqliteConnection.cursor()
    print("Connected to SQLite")
    sqlite_select_query = """SELECT * from databs where mem_id=1"""
    cursor.execute(sqlite_select_query)
    records = cursor.fetchall()
    for row in records:
        usermodel=row[1]
        low1 = row[2]
        low2 = row[3]
        low3=row[4]
        h1=row[5]
        h2=row[6]
        h3=row[7]


    cursor.close()
    flash('Image successfully uploaded and displayed below')
    frame = cv2.imread('static/uploads/'+usermodel)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of green color in HSV
    # lower_green = np.array([25, 52, 72])
    # upper_green = np.array([102, 255, 255])
    # lower_blue = np.array([110,50,50])
    # upper_blue = np.array([130,255,255])
    low_yell = np.array([low1, low2, low3])
    high_yell = np.array([h1, h2, h3])
    low_pin = np.array([42, 108, 90])
    high_pink = np.array([255, 255, 255])
    mask_white = cv2.inRange(hsv, low_yell, high_yell)
    mask_black = cv2.bitwise_not(mask_white)

    # converting mask_black to 3 channels
    W, L = mask_black.shape
    mask_black_3CH = np.empty((W, L, 3), dtype=np.uint8)
    mask_black_3CH[:, :, 0] = mask_black
    mask_black_3CH[:, :, 1] = mask_black
    mask_black_3CH[:, :, 2] = mask_black

    # cv2.imshow('orignal Image', frame)
    # cv2.imshow('mask_black',mask_black_3CH)

    dst3 = cv2.bitwise_and(mask_black_3CH, frame)
    # cv2.imshow('Pic+mask_inverse',dst3)

    # ///////
    W, L = mask_white.shape
    mask_white_3CH = np.empty((W, L, 3), dtype=np.uint8)
    mask_white_3CH[:, :, 0] = mask_white
    mask_white_3CH[:, :, 1] = mask_white
    mask_white_3CH[:, :, 2] = mask_white
    dst3_wh = cv2.bitwise_or(mask_white_3CH, dst3)
    design = cv2.imread('static/uploads/006276_1.jpg')
    design = cv2.resize(design, mask_black.shape[1::-1])
    # cv2.imshow('Shirt Design', design)

    design_mask_mixed = cv2.bitwise_or(mask_black_3CH, design)
    # cv2.imshow('design_mask_mixed',design_mask_mixed)

    final_mask_black_3CH = cv2.bitwise_and(design_mask_mixed, dst3_wh)

    outpath = "Output.jpg"
    # save the image
    cv2.imwrite('static/' + outpath, final_mask_black_3CH)
    filename = secure_filename(outpath)
    # cv2.imshow('Final Output',final_mask_black_3CH)

    # cv2.waitKey()

    # cv2.cv2.destroyAllWindows()  # Destroys all the windows created by imshow

    return render_template('index.html', filename=filename)

@app.route('/pic6/' ,methods=['POST'])
def pic6():
    sqliteConnection = sqlite3.connect('db_FINAL.db')
    cursor = sqliteConnection.cursor()
    print("Connected to SQLite")
    sqlite_select_query = """SELECT * from databs where mem_id=1"""
    cursor.execute(sqlite_select_query)
    records = cursor.fetchall()
    for row in records:
        usermodel=row[1]
        low1 = row[2]
        low2 = row[3]
        low3=row[4]
        h1=row[5]
        h2=row[6]
        h3=row[7]


    cursor.close()
    flash('Image successfully uploaded and displayed below')
    frame = cv2.imread('static/uploads/'+usermodel)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of green color in HSV
    # lower_green = np.array([25, 52, 72])
    # upper_green = np.array([102, 255, 255])
    # lower_blue = np.array([110,50,50])
    # upper_blue = np.array([130,255,255])
    low_yell = np.array([low1, low2, low3])
    high_yell = np.array([h1, h2, h3])
    low_pin = np.array([42, 108, 90])
    high_pink = np.array([255, 255, 255])
    mask_white = cv2.inRange(hsv, low_yell, high_yell)
    mask_black = cv2.bitwise_not(mask_white)

    # converting mask_black to 3 channels
    W, L = mask_black.shape
    mask_black_3CH = np.empty((W, L, 3), dtype=np.uint8)
    mask_black_3CH[:, :, 0] = mask_black
    mask_black_3CH[:, :, 1] = mask_black
    mask_black_3CH[:, :, 2] = mask_black

    # cv2.imshow('orignal Image', frame)
    # cv2.imshow('mask_black',mask_black_3CH)

    dst3 = cv2.bitwise_and(mask_black_3CH, frame)
    # cv2.imshow('Pic+mask_inverse',dst3)

    # ///////
    W, L = mask_white.shape
    mask_white_3CH = np.empty((W, L, 3), dtype=np.uint8)
    mask_white_3CH[:, :, 0] = mask_white
    mask_white_3CH[:, :, 1] = mask_white
    mask_white_3CH[:, :, 2] = mask_white
    dst3_wh = cv2.bitwise_or(mask_white_3CH, dst3)
    design = cv2.imread('static/uploads/005737_1.jpg')
    design = cv2.resize(design, mask_black.shape[1::-1])
    # cv2.imshow('Shirt Design', design)

    design_mask_mixed = cv2.bitwise_or(mask_black_3CH, design)
    # cv2.imshow('design_mask_mixed',design_mask_mixed)

    final_mask_black_3CH = cv2.bitwise_and(design_mask_mixed, dst3_wh)

    outpath = "Output.jpg"
    # save the image
    cv2.imwrite('static/' + outpath, final_mask_black_3CH)
    filename = secure_filename(outpath)
    # cv2.imshow('Final Output',final_mask_black_3CH)

    # cv2.waitKey()

    # cv2.cv2.destroyAllWindows()  # Destroys all the windows created by imshow

    return render_template('index.html', filename=filename)



@app.route('/display/<filename>')
def display_image(filename):
    # print('display_image filename: ' + filename)
    return redirect(url_for('static', filename=filename), code=301)


if __name__ == "__main__":
    app.run()