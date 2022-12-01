from tkinter import *
from tkinter import messagebox
import main
from random import choice, randint, shuffle
import json
Font = ("Combo", 20, "bold")




def adding():
    getting_name = name_entry.get()
    getting_goal = goal_entry.get()
    getting_arm = arm_entry.get()
    new_data = {getting_name: {
        "name": getting_name,
        "goal": getting_goal,
        "arm": getting_arm
    }
    }
    if len(getting_name) == 0 or len(getting_goal) == 0 or len(getting_arm) == 0:
        messagebox.showerror(title="Error ", message="Error you should fill all fields to save it ")

    else:
        try:
            with open("Gython.json", "r")as mydata:
                data = json.load(mydata)
        except FileNotFoundError:
            with open("Gython.json ", "w")as mydata:
                json.dump(new_data, mydata, indent=3)
        else:
            data.update(new_data)
            with open("Gython.json", "w")as mydata:
                json.dump(data, mydata, indent=4)
        finally:
            name_entry.delete(0, END)
            goal_entry.delete(0, END)
            arm_entry.delete(0, END)

    main.main(int(getting_goal), getting_arm)
    messagebox.showinfo (title = "Congratulations!", message = " You have already reached your goal! ")


def searching():
    getting_name = name_entry.get()
    try:
        with open("Gython.json", mode="r")as mydata:
            data = json.load(mydata)
    except FileNotFoundError:
        messagebox.showinfo(title="Warning", message="the file isn't here")
    else:
        if getting_name in data:
            getting_name_from_data_base = data[getting_name]["name"]
            messagebox.showinfo(title=getting_name, message=" Found already existing user! ")
        else:
            messagebox.showerror(title="error", message=f"there isn't information about {getting_name}!")
# ----------Window Settings --------


window = Tk()
window.title("Bicep Trainer")
window.config(padx=50, pady=50)
canvas = Canvas(width=200, height=100)

canvas.grid(row=0, column=1)
# whatev = PhotoImage(file = "./img/pose.jpg")
# canvas.create_image(500, 500, image=whatev)
# ----------------UI-------------
"""""labels"""
description = Label(text="Start your training!", font=("Combo", 20, "bold"))
description.grid(row=0, column=1)
name_label = Label(text="Name:", font=("Combo", 20, "bold"))
name_label.grid(row=1, column=0)
goal_label = Label(text="Goal:", font=Font)
goal_label.grid(row=3, column=0)
arm_label = Label(text="Which arm? ", font=Font)
arm_label.grid(row=4, column=0)

""""Buttons"""""
add = Button(text="Add", command=adding, width=36)
add.grid(row=5, column=1, columnspan=2)
search = Button(text="Search", command=searching)
search.grid(row=1, column=3)

""""Entries"""""

name_entry = Entry(width=35)
name_entry.grid(row=1, column=1, columnspan=2)
name_entry.focus()
goal_entry = Entry(width=35)
goal_entry.grid(row=3, column=1, columnspan=2)
arm_entry = Entry(width=35)
arm_entry.grid(row=4, column=1, columnspan=2)

window.mainloop()