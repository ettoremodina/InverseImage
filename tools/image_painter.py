"""
Simple RGBA Image Painter
A tkinter-based tool for creating RGBA images with a brush and color palette.
Outputs high-resolution images with transparent backgrounds.
"""

import tkinter as tk
from tkinter import colorchooser, filedialog, ttk
from PIL import Image, ImageDraw, ImageTk
import numpy as np

CANVAS_SIZE = 512
DEFAULT_BRUSH_SIZE = 15


class ImagePainter:
    
    def __init__(self, root):
        self.root = root
        self.root.title("RGBA Image Painter")
        
        self.image = Image.new('RGBA', (CANVAS_SIZE, CANVAS_SIZE), (0, 0, 0, 0))
        self.draw = ImageDraw.Draw(self.image)
        
        self.current_color = (255, 100, 50, 255)
        self.brush_size = DEFAULT_BRUSH_SIZE
        self.last_x = None
        self.last_y = None
        
        self.setup_ui()
        self.update_canvas()
    
    def setup_ui(self):
        main_frame = tk.Frame(self.root)
        main_frame.pack(padx=10, pady=10)
        
        # Canvas with checkerboard background (indicates transparency)
        canvas_frame = tk.Frame(main_frame, relief=tk.SUNKEN, borderwidth=2)
        canvas_frame.grid(row=0, column=0, rowspan=10, padx=(0, 10))
        
        self.canvas = tk.Canvas(canvas_frame, width=CANVAS_SIZE, height=CANVAS_SIZE, bg='#808080')
        self.canvas.pack()
        
        self.draw_checkerboard()
        
        self.canvas.bind('<B1-Motion>', self.paint)
        self.canvas.bind('<Button-1>', self.paint)
        self.canvas.bind('<ButtonRelease-1>', self.reset_last_pos)
        self.canvas.bind('<B3-Motion>', self.erase)
        self.canvas.bind('<Button-3>', self.erase)
        self.canvas.bind('<ButtonRelease-3>', self.reset_last_pos)
        
        # Controls
        controls_frame = tk.Frame(main_frame)
        controls_frame.grid(row=0, column=1, sticky='n')
        
        # Color preview
        tk.Label(controls_frame, text="Current Color:").pack(pady=(0, 5))
        self.color_preview = tk.Canvas(controls_frame, width=60, height=60, relief=tk.RAISED, borderwidth=2)
        self.color_preview.pack(pady=(0, 10))
        self.update_color_preview()
        
        # Color picker button
        tk.Button(controls_frame, text="Pick Color", command=self.pick_color, width=15).pack(pady=5)
        
        # Preset colors
        tk.Label(controls_frame, text="Preset Colors:").pack(pady=(15, 5))
        presets_frame = tk.Frame(controls_frame)
        presets_frame.pack()
        
        preset_colors = [
            ('#FF0000', 'Red'), ('#00FF00', 'Green'), ('#0000FF', 'Blue'),
            ('#FFFF00', 'Yellow'), ('#FF00FF', 'Magenta'), ('#00FFFF', 'Cyan'),
            ('#FFFFFF', 'White'), ('#000000', 'Black'), ('#FF8000', 'Orange'),
            ('#8000FF', 'Purple'), ('#00FF80', 'Mint'), ('#FF0080', 'Pink'),
        ]
        
        for i, (color, name) in enumerate(preset_colors):
            btn = tk.Button(presets_frame, bg=color, width=3, height=1,
                          command=lambda c=color: self.set_color_hex(c))
            btn.grid(row=i//4, column=i%4, padx=2, pady=2)
        
        # Brush size
        tk.Label(controls_frame, text="Brush Size:").pack(pady=(20, 5))
        self.brush_slider = tk.Scale(controls_frame, from_=1, to=50, orient=tk.HORIZONTAL,
                                     command=self.update_brush_size, length=150)
        self.brush_slider.set(DEFAULT_BRUSH_SIZE)
        self.brush_slider.pack()
        
        # Alpha slider
        tk.Label(controls_frame, text="Opacity:").pack(pady=(15, 5))
        self.alpha_slider = tk.Scale(controls_frame, from_=0, to=255, orient=tk.HORIZONTAL,
                                     command=self.update_alpha, length=150)
        self.alpha_slider.set(255)
        self.alpha_slider.pack()
        
        # Buttons
        tk.Button(controls_frame, text="Clear Canvas", command=self.clear_canvas, width=15).pack(pady=(20, 5))
        tk.Button(controls_frame, text="Save Image", command=self.save_image, width=15).pack(pady=5)
        tk.Button(controls_frame, text="Load Image", command=self.load_image, width=15).pack(pady=5)
        
        # Instructions
        instructions = tk.Label(controls_frame, text="Left click: Paint\nRight click: Erase",
                               font=('Arial', 9), fg='gray')
        instructions.pack(pady=(20, 0))
    
    def draw_checkerboard(self):
        checker_size = 16
        for i in range(0, CANVAS_SIZE, checker_size):
            for j in range(0, CANVAS_SIZE, checker_size):
                color = '#CCCCCC' if (i // checker_size + j // checker_size) % 2 == 0 else '#999999'
                self.canvas.create_rectangle(i, j, i + checker_size, j + checker_size,
                                            fill=color, outline='', tags='checker')
    
    def update_canvas(self):
        self.tk_image = ImageTk.PhotoImage(self.image)
        self.canvas.delete('image')
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image, tags='image')
        self.canvas.tag_raise('image')
    
    def update_color_preview(self):
        r, g, b, a = self.current_color
        hex_color = f'#{r:02x}{g:02x}{b:02x}'
        self.color_preview.delete('all')
        self.color_preview.create_rectangle(0, 0, 60, 60, fill=hex_color, outline='')
        if a < 255:
            self.color_preview.create_text(30, 30, text=f'{a}', fill='white' if r+g+b < 384 else 'black')
    
    def pick_color(self):
        color = colorchooser.askcolor(title="Choose Color")
        if color[0]:
            r, g, b = [int(c) for c in color[0]]
            self.current_color = (r, g, b, self.current_color[3])
            self.update_color_preview()
    
    def set_color_hex(self, hex_color):
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
        self.current_color = (r, g, b, self.current_color[3])
        self.update_color_preview()
    
    def update_brush_size(self, val):
        self.brush_size = int(val)
    
    def update_alpha(self, val):
        r, g, b, _ = self.current_color
        self.current_color = (r, g, b, int(val))
        self.update_color_preview()
    
    def paint(self, event):
        x, y = event.x, event.y
        r = self.brush_size // 2
        
        if self.last_x is not None and self.last_y is not None:
            self.draw.line([(self.last_x, self.last_y), (x, y)], 
                          fill=self.current_color, width=self.brush_size)
        
        self.draw.ellipse([x - r, y - r, x + r, y + r], fill=self.current_color)
        
        self.last_x = x
        self.last_y = y
        self.update_canvas()
    
    def erase(self, event):
        x, y = event.x, event.y
        r = self.brush_size // 2
        
        if self.last_x is not None and self.last_y is not None:
            self.draw.line([(self.last_x, self.last_y), (x, y)], 
                          fill=(0, 0, 0, 0), width=self.brush_size)
        
        self.draw.ellipse([x - r, y - r, x + r, y + r], fill=(0, 0, 0, 0))
        
        self.last_x = x
        self.last_y = y
        self.update_canvas()
    
    def reset_last_pos(self, event):
        self.last_x = None
        self.last_y = None
    
    def clear_canvas(self):
        self.image = Image.new('RGBA', (CANVAS_SIZE, CANVAS_SIZE), (0, 0, 0, 0))
        self.draw = ImageDraw.Draw(self.image)
        self.update_canvas()
    
    def save_image(self):
        file_path = filedialog.asksaveasfilename(
            defaultextension='.png',
            filetypes=[('PNG files', '*.png'), ('All files', '*.*')],
            initialdir='../images'
        )
        if file_path:
            self.image.save(file_path, 'PNG')
            print(f"Image saved to: {file_path}")
    
    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[('PNG files', '*.png'), ('All files', '*.*')],
            initialdir='../images'
        )
        if file_path:
            loaded = Image.open(file_path).convert('RGBA')
            loaded = loaded.resize((CANVAS_SIZE, CANVAS_SIZE), Image.Resampling.LANCZOS)
            self.image = loaded
            self.draw = ImageDraw.Draw(self.image)
            self.update_canvas()
            print(f"Image loaded from: {file_path}")


def main():
    root = tk.Tk()
    root.resizable(False, False)
    app = ImagePainter(root)
    root.mainloop()


if __name__ == '__main__':
    main()
