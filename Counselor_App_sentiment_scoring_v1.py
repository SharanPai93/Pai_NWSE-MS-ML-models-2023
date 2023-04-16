#Counselor App with NLP model support
#Implemented using Tkinter APIs and BERT model
#Sharan Pai, April 2023

from transformers import pipeline
from tkinter import *
from tkinter.ttk import *
from tkinter import messagebox
from time import strftime
import time
import threading
from tkinter import scrolledtext
from PIL import ImageTk, Image

time = time.strftime("%I:%M %D")

#Install this BERT model from Huggingface community framework
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
        i = IntVar()

        self.root.title("Check-in app")
        self.root.geometry("1000x700")
        
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
            score = classified_text['score']
            emotion_score = [str(emotion), float(score)]
            
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
                
                if text == "":
                    reasonName = "Empty"

                else:
                    if text.isalpha():
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

        
        # define a function for 1st toplevel
        # which is associated with self.root window.
        def change_view_counselor():
            # Create widget
            counselor_view = Toplevel(self.root)

            self.root.withdraw()

            login_button = Button(counselor_view, text='Login')

            login_button.grid(row=0,column=0,sticky='W')
            
            contact_lbl = Label(counselor_view, text="  Student's guardian contact : ",
                                    font=('Times New Roman', 12))

            contact_lbl.grid(row=1, column=9, columnspan=4)

            span_lbl = Label(counselor_view, text=" 111-111-1111",
                             font=('Times New Roman',12))

            span_lbl.grid(row=1, column=14, columnspan=2)

            grade_lbl = Label(counselor_view, text=f"Student's grade: {gradeLevel.get()}",
                              font=('Times New Roman',12))

            grade_lbl.grid(row=2,column=0,columnspan=3,sticky='W')

            def clear_counselor_all():
                notes_box.delete("1.0","end-1c")
                if i.get() == 1:
                    i.set(None)
                elif i.get() == 2:
                    i.set(None)
                elif i.get() == 3:
                    i.set(None)
            
            def check_radio_severity():
                if i.get() == 1:
                    messagebox.showinfo("Successfully Saved!", "Your data has been successfully saved.")
                    clear_counselor_all()
                elif i.get() == 2:
                    messagebox.showinfo("Successfully Saved!", "Your data has been successfully saved.")
                    clear_counselor_all()
                elif i.get() == 3:
                    messagebox.showinfo("Successfully Saved!", "Your data has been successfully saved.")
                    clear_counselor_all()
                else:
                    messagebox.showerror("Value Error", "A level of severity must be chosen")
                    
            def lift_and_destroy():
                self.root.update()
                self.root.deiconify()
                counselor_view.destroy()
            
            def update_values(index, value, op):
                #print("\nChanged Combo value is: " + combo.get())
                
                if list_of_students.get() == "Student A":
                    span_lbl.configure(text="111-111-1111")
                elif list_of_students.get() == "Student B":
                    span_lbl.configure(text="222-222-2222")
                elif list_of_students.get() == "Student C":
                    span_lbl.configure(text="333-333-3333")
                
                
            # Define title for window
            counselor_view.title("Counselor view")
             
            # specify size
            counselor_view.geometry('725x550')
            
            list_students_label = Label(counselor_view, text=" List of students: ",
                                        font=('Times New Roman', 12))

            list_students_label.grid(row=1, column=0, columnspan=2,sticky="W")

            list_of_students = Combobox(counselor_view, font=('Times New Roman', 12),
                                        textvar=s)

            list_of_students['values'] = ("Student A", "Student B", "Student C")

            list_of_students.current(0)

            list_of_students.grid(row=1, column=2,columnspan=4)

            sentences = []
            sentences = boxText.get('1.0', 'end-1c')
            ##############################
            #Function to perform sentiment scoring on the input sentences
            #use input as sentences and output as score    
            
            score = f"{classify_emotion(sentences, 'score+emotion')[0]},\
 {round(classify_emotion(sentences, 'score+emotion')[1], 2)}"
            
            ##############################
            sentiment_changer = Label(counselor_view, text=f' Sentiment score\
output: {score}', font=('Times New Roman', 12))

            sentiment_changer.grid(row=3, column=0,columnspan=3,sticky='W')

            sentiment_entry = Entry(counselor_view)

            #Perform sentiment analysis 
            #sentiment_entry.insert(0,'0.5')
            #sentiment_entry.insert(0, str(score))
            ##########################################
            sentiment_entry.grid(row=3,column=3,sticky='W')
            
            canvas = Canvas(counselor_view, width=100, height=30)

            canvas.grid(row=5,column=0)          
            note_label = Label(counselor_view, text="Notes: ",
                               font=('Times New Roman', 12))

            #note_label.grid(row=400,column=0,sticky="W")
            
            #notes_box = scrolledtext.ScrolledText(counselor_view, wrap=WORD,
                                           # width=50, height=5,
                                           # font=("Times New Roman", 12))

            #notes_box.grid(row=401,column=0,columnspan=10,sticky="W")

            severity_label = Label(counselor_view, text="Severity of scenario: ",
                                   font=("Times New Roman", 12))

            #severity_label.grid(row=41,column=0,pady=2,sticky="W",columnspan=3)
            
            severity_radio1 = Radiobutton(counselor_view,text="Low",
                                          value=1,variable=i)
            
            severity_radio2 = Radiobutton(counselor_view,text="Medium",
                                          value=2,variable=i)

            severity_radio3 = Radiobutton(counselor_view,text="High",
                                          value=3,variable=i)

            Separator(counselor_view, orient=VERTICAL).grid(column=0, row=6, rowspan=1, sticky='nsW')

            severity_radio1.grid(row=402,column=1,sticky="W")

            severity_radio2.grid(row=402,column=3,sticky="W")

            severity_radio3.grid(row=402,column=4,sticky="W")

            save_button = Button(counselor_view, text="Save info for student", command=check_radio_severity)
            
            save_button.grid(row=403,column=0,columnspan=4,pady=5, sticky="W")
            messagebox.showinfo(title='Output',message=sentences)
            
            # Create Exit button
            exit_counselor_view = Button(counselor_view, text = "Exit counselor view",
                             command = lift_and_destroy)
             
            exit_counselor_view.grid(row=404,column=0,pady=10,columnspan=2,
                                     sticky='W')

            s.trace('w',update_values)
            
            # Display until closed manually
            counselor_view.mainloop()
        def click():
            sentences = []
            score = 0
            sentences = boxText.get('1.0', 'end-1c')
            score = 2
            #print(sentences)
        
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
                clear_all()
                messagebox.showinfo("Submit Success", "You have submitted the data successfully!\
                                     You can safely exit this window now.")
                print(time)
        
        #Name entry box
        name = Entry(self.root, width=30, font=('Times New Roman', 12))

        nameSentence = Label(self.root, text="Enter Student's name (Last name, First name):",
                             font=("Times New Roman", 12))

        nameSentence.grid(row=0, column=0,columnspan=12,sticky="W")

        name.grid(row=0,column=7, columnspan=5, sticky="W")

        gradeLevelLabel = Label(self.root, text="Grade Level:",
                                font=("Times New Roman", 12))
        gradeLevel = Entry(self.root, width=3, font=('Times New Roman', 12))

        gradeLevelLabel.grid(row=1,column=0, columnspan=2, sticky="W")

        gradeLevel.grid(row=1, column=2, sticky="W")

        boxText = scrolledtext.ScrolledText(self.root, wrap=WORD,
                                            width=120, height=30,
                                            font=("Times New Roman", 12))

        #boxText.pack(fill=BOTH, side=LEFT)
        boxText.grid(row=2, rowspan=24, column=0, columnspan=19, pady=2)
        '''
        boxText.insert(INSERT,
                       """\
                            Sample Text Sample Text Sample Text Sample Text Sample Text Sample Text Sample Text 
                            Testing Sample Text Sample Text Sample Text Sample Text Sample Text Sample Text 
                            Sample Text Sample Text Sample Text Sample Text Sample Text Sample Text Sample Text
                            Sample Text Sample Text Sample Text Sample Text Sample Text Sample Text
                            Sample Text Sample Text Sample Text Sample Text Sample Text Sample Text !!!
                            """)
        '''
        #boxText.configure(state='disabled')
        
        #submitButton = Button(self.root, text="Save", command = submit_student)
        #submitButton.grid(row=26,column=0, columnspan=2, sticky="W",pady=3)
        
        #Button(self.root, text = "Save", width=8, command=click).grid(row=26,column=0,sticky="W",pady=3)
         
        # Create button to open toplevel1
        switch_button = Button(self.root, text = "Perform Sentiment Analysis",
                        command = change_view_counselor)
        #button.pack()
         
        # position the button
        switch_button.grid(row=27,column=0,columnspan=4,sticky="W",pady=3)
        
        # Display until closed manually
        self.root.mainloop()
        

app = StudentCounselorApp()
