import tkinter as tk
import math

# Class to represent the L-System
# Instead of generating the whole string, this class receives the canvas and draws directly on it recursively
class LSystem:
    def __init__(self, number):
        self.line_length = 5
        self.starting_angle = 0
        match number:
            case 0:
                self.axiom = "F+F+F+F"
                self.rule = "F+F-F-FF+F+F-F"
                self.angle = math.pi / 2
            case 1:
                self.axiom = "F++F++F"
                self.rule = "F+F--F+F"
                self.angle = math.pi / 3
            case 2:
                self.axiom = "F"
                self.rule = "F[+F]F[-F]F"
                self.angle = math.pi / 7
            case 3:
                self.axiom = "F"
                self.rule = "FF+[+F-F-F]-[-F+F+F]"
                self.angle = math.pi / 8

    # Draw into canvas
    def draw(self, canvas, iterations, start_x, start_y, start_angle):
        self.max_depth = iterations - 1
        self.current_y = start_y
        self.current_x = start_x
        self.current_angle = start_angle

        for cmd in self.axiom:
            # Replace the axiom with the rule
            if cmd == "F":
                self.draw_recursive(canvas, 0)
            # Adjust angle
            elif cmd == "+":
                self.current_angle = (self.current_angle + self.angle) % (math.pi * 2)
            elif cmd == "-":
                self.current_angle -= self.angle
                if self.current_angle < 0:
                    self.current_angle += math.pi * 2

    def draw_recursive(self, canvas, depth):
        stack = []
        for cmd in self.rule:
            if cmd == "F":
                # If depth is reached, start drawing
                if depth == self.max_depth:
                    new_x = self.current_x + self.line_length * math.cos(self.current_angle) # New x after moving
                    new_y = self.current_y + self.line_length * math.sin(self.current_angle) # New y after moving

                    # Check if new position is within canvas bounds
                    if new_x > canvas.winfo_width() or new_x < 0 or new_y > canvas.winfo_height() or new_y < 0:
                        continue

                    canvas.create_line(self.current_x, self.current_y, new_x, new_y) # Draw line on canvas
                    self.current_x, self.current_y = new_x, new_y

                # If not, replace F with rule again
                else:
                    self.draw_recursive(canvas, depth + 1)

            elif cmd == "+":
                self.current_angle = (self.current_angle + self.angle) % (math.pi * 2)

            elif cmd == "-":
                self.current_angle -= self.angle
                if self.current_angle < 0:
                    self.current_angle += math.pi * 2

            # Push checkpoint to the stack
            elif cmd == "[":
                stack.append((self.current_x, self.current_y, self.current_angle))

            # Retrieve last checkpoint from the stack
            elif cmd == "]":
                self.current_x, self.current_y, self.current_angle = stack.pop()


class App:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("L-Systems")

        self.canvas_frame = tk.Frame(self.root)
        self.control_frame = tk.Frame(self.root)

        self.canvas_frame.pack(side="left", padx=10, pady=10, fill=tk.BOTH, expand=True)
        self.control_frame.pack(side="right", padx=10, pady=10, fill=tk.Y)

        # Canvas setup
        self.canvas = tk.Canvas(self.canvas_frame, bg="white", width=800, height=800)
        self.canvas.pack(fill=tk.BOTH, expand=True)  # Allow canvas to resize

        self.l_system = LSystem(0)  # Default L-System

        tk.Label(self.control_frame, text="Pick L-System").pack(side=tk.TOP, pady=5)

        # L-System picker
        for i in range(4):
            btn = tk.Button(
                self.control_frame,
                text="L-System " + str(i + 1)
            )
            btn.config(command=lambda num=i: self.switch_system(num))
            btn.pack(side=tk.TOP, pady=2)

        # Nesting level slider
        self.nesting_level = tk.IntVar()
        self.nesting_level.set(3)
        tk.Label(self.control_frame, text="Nesting Level").pack(side=tk.TOP, pady=5)
        tk.Scale(
            self.control_frame,
            variable=self.nesting_level,
            from_=1,
            to=10,
            orient=tk.HORIZONTAL,
        ).pack(side=tk.TOP)

        # Line length (size)
        self.line_length = tk.IntVar()
        self.line_length.set(5)
        tk.Label(self.control_frame, text="Line Length").pack(side=tk.TOP, pady=5)
        tk.Scale(
            self.control_frame,
            variable=self.line_length,
            from_= 1,
            to=20,
            orient=tk.HORIZONTAL,
        ).pack(side=tk.TOP)

        # Starting coordinates
        self.start_x = tk.IntVar()
        self.start_x.set(200)
        self.start_y = tk.IntVar()
        self.start_y.set(200)
        tk.Label(self.control_frame, text="Starting X").pack(side=tk.TOP, pady=5)
        tk.Entry(self.control_frame, textvariable=self.start_x).pack(side=tk.TOP)
        tk.Label(self.control_frame, text="Starting Y").pack(side=tk.TOP, pady=5)
        tk.Entry(self.control_frame, textvariable=self.start_y).pack(side=tk.TOP)

        # Starting angle
        self.start_angle = tk.IntVar()
        self.start_angle.set(0)
        tk.Label(self.control_frame, text="Starting Angle").pack(side=tk.TOP, pady=5)
        tk.Entry(self.control_frame, textvariable=self.start_angle).pack(side=tk.TOP)

        # Draw button
        draw_btn = tk.Button(
            self.control_frame,
            text="Draw",
            command=self.draw,
        )
        draw_btn.pack(side=tk.TOP, pady=5)

        # Current L-System info
        l_system_info = tk.Frame(self.control_frame)
        tk.Label(l_system_info, text="Current L-System:").grid(row=0, column=0, columnspan=2)
        tk.Label(l_system_info, text="Axiom:").grid(row=1, column=0)
        self.l_system_axiom = tk.Label(l_system_info, text=self.l_system.axiom)
        self.l_system_axiom.grid(row=1, column=1)
        tk.Label(l_system_info, text="Rule:").grid(row=2, column=0)
        self.l_system_rule = tk.Label(l_system_info, text="F-> " + self.l_system.rule)
        self.l_system_rule.grid(row=2, column=1)
        tk.Label(l_system_info, text="Angle:").grid(row=3, column=0)
        self.l_system_angle = tk.Label(l_system_info, text="{:.2f}".format(math.degrees(self.l_system.angle)))
        self.l_system_angle.grid(row=3, column=1)
        l_system_info.pack(side=tk.BOTTOM)

    def switch_system(self, num):
        self.l_system = LSystem(num)
        self.l_system_axiom.config(text=self.l_system.axiom)
        self.l_system_rule.config(text="F-> " + self.l_system.rule)
        self.l_system_angle.config(text="{:.2f}".format(math.degrees(self.l_system.angle)))

    def draw(self):
        self.canvas.delete("all")
        self.l_system.line_length = self.line_length.get() # Update line length
        start_angle = math.radians(self.start_angle.get()) # Convert to radians
        self.l_system.draw(self.canvas, self.nesting_level.get(), self.start_x.get(), self.start_y.get(), start_angle)

    def run(self):
        self.root.mainloop()


# Run the tkinter application
app = App()
app.run()