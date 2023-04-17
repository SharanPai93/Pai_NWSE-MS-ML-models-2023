#Counselor App with NLP model support
#Implemented using Tkinter APIs and BERT model
#Sharan Pai, April 2023
#NOTES:
#When run, the app sometimes opens in a window not seen, and to view it, press alt+tab
#Occasionally, errors might show, but they (usually) do not affect the performance of the app

from tkinter import *
from tkinter.ttk import *
from tkinter import messagebox
from time import strftime
import time
import threading
from tkinter import scrolledtext
from PIL import ImageTk, Image
import random

#Might take a bit of time to import
from transformers import pipeline

time = time.strftime("%I:%M %D")

emotion_classifier = pipeline("text-classification",
                              model='bhadresh-savani/distilbert-base-uncased-emotion',
                      top_k=True)

class StudentCounselorApp(threading.Thread):
    
    def __init__(self):
        threading.Thread.__init__(self)
        self.start()
        
    
    def run(self):
        self.root = Tk()
        s = StringVar()
        self.box_text_list = []
        i = IntVar()
        self.root['background'] = '#ADD8E6'

        self.root.title("Counseling App")
        self.root.geometry("1000x800")

        self.student_dict = {}
        self.student_list = []
        
        def clear_all():
            name.delete(0, END)
            gradeLevel.delete(0, END)
            boxText.delete("1.0", "end")

        def clear_specific(name_of_box):
            name_of_box.delete(0, END)

        def classify_emotion(text,type_out=''):
            classified_text = emotion_classifier(text)
            classified_text = classified_text[0][0]
            emotion = classified_text['label']
            score = round(classified_text['score'],2)
            emotion_score = f'{str(emotion)}, {float(score)}'
            
            if type_out == '':
                raise ValueError('type_out not specified')

            elif type_out == ' ':
                messagebox.showerror(title='No writing done by student', message='The student\
did not write anything down to classify.')
                NaN = 'NaN'
                return NaN
            
            elif type_out == 'emotion':
                return emotion

            elif type_out == 'score':
                return score

            elif (type_out == 'emotion+score' or type_out == 'score+emotion'):
                return emotion_score

            else:
                raise ValueError('Incorrect value of type_out')

        def check(input_text, type):
            if type == "any":
                textBox = input_text.get("1.0",'end-1c')

                if textBox == "":
                    messagebox.showerror("Value Error", "Writing box must be filled out")
                    return 0
                
                else:
                    return 1

            if type == "strLetters":
                reasonName = ""
                text = input_text.get()
                empty_text_list = []
                real_text_list = [str(i) for i in text]

                if text == "":
                    reasonName = "Empty"

                else:
                    for i in text:
                        if i == ' ':
                            empty_text_list.append(i)

                    if len(empty_text_list) == len(real_text_list):
                        reasonName = "Empty"
                        
                    else:
                        for i in text:
                            if i.isalpha() or i == ' ':
                                pass
                            else:
                                reasonName = "nonLetters"

                if reasonName == "":
                    return 1
                
                else:
                    if reasonName == "Empty":
                        messagebox.showerror("Value Error", "Name box must be filled out")
                    elif reasonName == "nonLetters":
                        messagebox.showerror("Value Error", "All characters in name box must be letters")
                    return 0

            elif type == "strNumbers":
                reasonGrade = ""
                gradeNum = input_text.get()
                if gradeNum == "":
                    reasonGrade = "Empty"
                    messagebox.showerror("Value Error", "Grade level must be filled")
                    return 0
                try:
                    gradeNum = float(input_text.get())
                except ValueError:
                    messagebox.showerror("Value Error", "Grade level cannot be a letter")
                    clear_specific(gradeLevel)
                    return 0
                
                
                else:
                    if float(gradeNum).is_integer():
                        if 0 < int(gradeNum) <= 12:
                            pass
                        else:
                            reasonGrade = "greaterLess12"
                    else:
                        reasonGrade = "nonIntegers"
            
            
                if reasonGrade == "":
                    return 1
            
                else:                
                    if reasonGrade == "nonIntegers":
                        messagebox.showerror("Value Error", "Grade level must be integers")
                    elif reasonGrade == "greaterLess12":
                        messagebox.showerror("Value Error", "Grade level must be greater than 0 and less or equal to 12")
                    return 0

        
        #Define a function for Counselor view (Toploevel)
        def change_view_counselor():
            # Create Counselor view widget
            counselor_view = Toplevel(self.root)
            counselor_view['background'] = 
            self.root.withdraw()

            def login():
                messagebox.showinfo(title='Success!',
                                    message='You have successfully logged in!')
            
            login_button = Button(counselor_view, text='Login', command=login)

            login_button.grid(row=0,column=0,sticky='W')
            
            contact_lbl = Label(counselor_view, text="Student's guardian contact is: ",
                                    font=('Times New Roman', 12))

            contact_lbl.grid(row=1, column=6, columnspan=5)

            span_lbl = Label(counselor_view, text="",
                             font=('Times New Roman',12))

            span_lbl.grid(row=1, column=11, columnspan=3)

            grade_lbl = Label(counselor_view, text="Student's grade: ",
                              font=('Times New Roman',12))

            span_grade = Label(counselor_view, text='',
                               font=('Times New Roman', 12))

            grade_lbl.grid(row=2,column=0,columnspan=3,sticky='W')

            span_grade.grid(row=2, column=3,sticky='W')

            list_students_label = Label(counselor_view, text="List of students: ",
                                        font=('Times New Roman', 12))

            list_students_label.grid(row=1, column=0, columnspan=3,sticky="W")

            list_of_students = Combobox(counselor_view, font=('Times New Roman', 12),
                                        textvar=s)

            list_of_students['values'] = [str(i) for i in self.student_list]

            list_of_students.current(0)

            list_of_students.grid(row=1, column=1,columnspan=5)

            sentiment_score = Label(counselor_view, text='Sentiment score of student:',
                                    font=('Times New Roman', 12))

            span_sentiment = Label(counselor_view, text='',
                                   font=('Times New Roman', 12))

            sentiment_score.grid(row=3,column=0,columnspan=4,sticky='W')

            span_sentiment.grid(row=3,column=4,columnspan=2,sticky='W')

            def update_score(index, value, op):
                name_index = self.student_list.index(str(list_of_students.get()))
                sentences = self.box_text_list[name_index]
                score = classify_emotion(sentences, 'score+emotion')
                span_sentiment.configure(text=f'{score}')
            
            def update_values(index, value, op):
                phone_list = []
                for i in range(1,11):
                    num = random.randrange(1,9)
                    if (i == 4 or i == 7):
                        phone_list.append('-')
                    phone_list.append(num)

                phone_num = ''.join([str(i) for i in phone_list])
                
                span_lbl.configure(text=f"{phone_num}")
            
            def update_student_grade(index, value, op):
                students_name = list_of_students.get()
                students_grade = self.student_dict[str(students_name)]
                span_grade.configure(text=f"{students_grade}")
            
            
            def clear_counselor_all():
                notes_box.delete("1.0","end-1c")
                if i.get() == 1:
                    i.set(None)
                elif i.get() == 2:
                    i.set(None)
                elif i.get() == 3:
                    i.set(None)
            
            def check_radio_severity():
                if (i.get() == 1 or i.get() == 2 or i.get() == 3):
                    messagebox.showinfo("Successfully Saved!", "Your data has been successfully saved.")
                    clear_counselor_all()
                else:
                    messagebox.showerror("Value Error", "A level of severity must be chosen")
                    
            def lift_and_destroy():
                self.root.update()
                self.root.deiconify()
                counselor_view.destroy()
                
                
            # Define title for window
            counselor_view.title("Sentiment Analysis")
             
            # specify size
            counselor_view.geometry('800x700')

            
            note_label = Label(counselor_view, text="Notes: ",
                               font=('Times New Roman', 12))

            note_label.grid(row=5,column=0,columnspan=1,sticky="W")
            
            notes_box = scrolledtext.ScrolledText(counselor_view, wrap=WORD,
                                            width=50, height=5,
                                            font=("Times New Roman", 12))

            notes_box.grid(row=6,column=0,columnspan=12,sticky="W")

            severity_label = Label(counselor_view, text="Severity of scenario: ",
                                   font=("Times New Roman", 12))

            severity_label.grid(row=7,column=0,pady=2,sticky="W",columnspan=3)
            
            severity_radio1 = Radiobutton(counselor_view,text="Low",
                                          value=1,variable=i)
            
            severity_radio2 = Radiobutton(counselor_view,text="Medium",
                                          value=2,variable=i)

            severity_radio3 = Radiobutton(counselor_view,text="High",
                                          value=3,variable=i)

            #Separator(counselor_view, orient=VERTICAL).grid(column=0, row=6, rowspan=1, sticky='nsW')

            severity_radio1.grid(row=8,column=1,sticky="W")

            severity_radio2.grid(row=8,column=3,columnspan=1,sticky="W")

            severity_radio3.grid(row=8,column=5,sticky="W")

            save_button = Button(counselor_view, text="Save info for student", command=check_radio_severity)
            
            save_button.grid(row=9,column=0,columnspan=4,pady=5, sticky="W")

            
            #Create Exit button
            exit_counselor_view = Button(counselor_view, text = "Exit counselor view",
                             command = lift_and_destroy)
             
            exit_counselor_view.grid(row=10,column=0,pady=10,columnspan=4,
                                     sticky='W')
            
            s.trace('w',update_values)
            s.trace('w',update_student_grade)
            s.trace('w',update_score)
            
            
            # Display until closed manually
            counselor_view.mainloop()

        
        def submit_student():
            
            value_check_name = check(name, "strLetters")
            value_check_grade = check(gradeLevel,"strNumbers")
            value_check_box = check(boxText, "any")
            
            if value_check_name == 0:
                check(name,"strLetters")

            elif value_check_grade == 0:
                check(gradeLevel,"strNumbers")

            elif value_check_box == 0:
                check(boxText, "any")
                
                
            elif (value_check_name == 1 and value_check_grade == 1 and value_check_box == 1):
                student_name = name.get()
                student_grade = gradeLevel.get()
                student_boxtext = boxText.get('1.0', 'end-1c')
                self.student_list.append(str(student_name))
                self.box_text_list.append(str(student_boxtext))
                self.student_dict[student_name] = student_grade
                    
                clear_all()
                messagebox.showinfo("Submit Success", "You have submitted the data successfully!\
                                     You can safely exit this window now.")
                print(f'Date of student entry: {time}')
        
        #Name entry box
        name = Entry(self.root, width=30, font=('Times New Roman', 12))

        nameSentence = Label(self.root, text="Student name:",
                             font=("Times New Roman", 12))

        nameSentence.grid(row=0, column=0,columnspan=3,sticky="W")

        name.grid(row=0,column=1, columnspan=6, sticky="W")

        gradeLevelLabel = Label(self.root, text="Grade Level:",
                                font=("Times New Roman", 12))
        gradeLevel = Entry(self.root, width=3, font=('Times New Roman', 12))

        gradeLevelLabel.grid(row=1,column=0, columnspan=2, sticky="W")

        gradeLevel.grid(row=1, column=1, sticky="W")

        boxText = scrolledtext.ScrolledText(self.root, wrap=WORD,
                                            width=120, height=30,
                                            font=("Times New Roman", 12))

        boxText.grid(row=2, rowspan=24, column=0, columnspan=24, pady=2)

        submitButton = Button(self.root, text="Submit", command = submit_student)

        submitButton.grid(row=26,column=0, columnspan=1, sticky="W",pady=3)
         
        # Create button to open toplevel1
        switch_button = Button(self.root, text = "Switch to counselor view",
                        command = change_view_counselor)
        #button.pack()
         
        # position the button
        switch_button.grid(row=27,column=0,columnspan=5,sticky="W",pady=3)
        
        # Display until closed manually
        self.root.mainloop()
        

app = StudentCounselorApp()
