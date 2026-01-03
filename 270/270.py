import tkinter as tk
import time
import json

results = {
    "start_time": None,
    "reaction_time": None,
    "feedback_clicked": False,
    "user_notes": ""
}

def show_feedback():
    global feedback_start
    feedback_label.config(text=" بازخورد سبک: عملکرد شما عالی بود!")
    feedback_start = time.time()

def user_clicked():
    end = time.time()
    results["reaction_time"] = round(end - feedback_start, 2)
    results["feedback_clicked"] = True
    results["user_notes"] = note_box.get("1.0", tk.END).strip()
    
    with open("ui_test_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    root.destroy()

root = tk.Tk()
root.title("تست UI با کاربران نهایی")
root.geometry("400x300")

results["start_time"] = time.time()

feedback_label = tk.Label(root, text="", font=("Arial", 14))
feedback_label.pack(pady=20)

btn_show = tk.Button(root, text="نمایش بازخورد", command=show_feedback)
btn_show.pack()

btn_accept = tk.Button(root, text="کاربر بازخورد را فهمیدم", command=user_clicked)
btn_accept.pack(pady=10)

note_box = tk.Text(root, height=4, width=40)
note_box.insert("1.0", "نظر کاربر را اینجا بنویسید...")
note_box.pack(pady=10)

root.mainloop()
