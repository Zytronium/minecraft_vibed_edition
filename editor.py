#!/usr/bin/env python3
"""
Minecraft: Vibed Edition - Data Editor
A GUI tool for editing blocks, items, and crafting recipes
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import json
import os
from typing import Dict, List, Any

class GameDataEditor:
    def __init__(self, root):
        self.root = root
        self.root.title("Minecraft: Vibed Edition - Data Editor")
        self.root.geometry("1200x800")

        # File paths
        self.blocks_file = "src/blocks.json"
        self.items_file = "src/items.json"
        self.crafting_file = "src/crafting.json"

        # Data storage
        self.blocks_data = {"blocks": []}
        self.items_data = {"items": []}
        self.crafting_data = {"recipes": []}

        # Load data
        self.load_all_data()

        # Create UI
        self.create_ui()

    def load_all_data(self):
        """Load all JSON data files"""
        try:
            if os.path.exists(self.blocks_file):
                with open(self.blocks_file, 'r') as f:
                    self.blocks_data = json.load(f)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load blocks: {e}")

        try:
            if os.path.exists(self.items_file):
                with open(self.items_file, 'r') as f:
                    self.items_data = json.load(f)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load items: {e}")

        try:
            if os.path.exists(self.crafting_file):
                with open(self.crafting_file, 'r') as f:
                    self.crafting_data = json.load(f)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load recipes: {e}")

    def save_blocks(self):
        """Save blocks to JSON file"""
        try:
            os.makedirs(os.path.dirname(self.blocks_file), exist_ok=True)
            with open(self.blocks_file, 'w') as f:
                json.dump(self.blocks_data, f, indent=2)
            messagebox.showinfo("Success", "Blocks saved successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save blocks: {e}")

    def save_items(self):
        """Save items to JSON file"""
        try:
            os.makedirs(os.path.dirname(self.items_file), exist_ok=True)
            with open(self.items_file, 'w') as f:
                json.dump(self.items_data, f, indent=2)
            messagebox.showinfo("Success", "Items saved successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save items: {e}")

    def save_crafting(self):
        """Save recipes to JSON file"""
        try:
            os.makedirs(os.path.dirname(self.crafting_file), exist_ok=True)
            with open(self.crafting_file, 'w') as f:
                json.dump(self.crafting_data, f, indent=2)
            messagebox.showinfo("Success", "Recipes saved successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save recipes: {e}")

    def create_ui(self):
        """Create the main UI"""
        # Create notebook (tabs)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=5, pady=5)

        # Create tabs
        self.blocks_tab = ttk.Frame(self.notebook)
        self.items_tab = ttk.Frame(self.notebook)
        self.crafting_tab = ttk.Frame(self.notebook)

        self.notebook.add(self.blocks_tab, text="Blocks")
        self.notebook.add(self.items_tab, text="Items")
        self.notebook.add(self.crafting_tab, text="Crafting")

        # Setup each tab
        self.setup_blocks_tab()
        self.setup_items_tab()
        self.setup_crafting_tab()

    def setup_blocks_tab(self):
        """Setup the blocks editor tab"""
        # Left panel - list of blocks
        left_frame = ttk.Frame(self.blocks_tab)
        left_frame.pack(side='left', fill='both', expand=False, padx=5, pady=5)

        ttk.Label(left_frame, text="Blocks:", font=('Arial', 12, 'bold')).pack()

        # Search
        search_frame = ttk.Frame(left_frame)
        search_frame.pack(fill='x', pady=5)
        ttk.Label(search_frame, text="Search:").pack(side='left')
        self.block_search = ttk.Entry(search_frame)
        self.block_search.pack(side='left', fill='x', expand=True, padx=5)
        self.block_search.bind('<KeyRelease>', lambda e: self.filter_blocks())

        # Listbox
        list_frame = ttk.Frame(left_frame)
        list_frame.pack(fill='both', expand=True)

        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side='right', fill='y')

        self.blocks_list = tk.Listbox(list_frame, yscrollcommand=scrollbar.set, width=30)
        self.blocks_list.pack(side='left', fill='both', expand=True)
        scrollbar.config(command=self.blocks_list.yview)
        self.blocks_list.bind('<<ListboxSelect>>', self.on_block_select)

        # Buttons
        btn_frame = ttk.Frame(left_frame)
        btn_frame.pack(fill='x', pady=5)
        ttk.Button(btn_frame, text="New Block", command=self.new_block).pack(fill='x', pady=2)
        ttk.Button(btn_frame, text="Delete Block", command=self.delete_block).pack(fill='x', pady=2)
        ttk.Button(btn_frame, text="Save Blocks", command=self.save_blocks).pack(fill='x', pady=2)

        # Right panel - block editor
        right_frame = ttk.Frame(self.blocks_tab)
        right_frame.pack(side='right', fill='both', expand=True, padx=5, pady=5)

        # Create scrollable canvas for form
        canvas = tk.Canvas(right_frame)
        scrollbar = ttk.Scrollbar(right_frame, orient="vertical", command=canvas.yview)
        self.block_form_frame = ttk.Frame(canvas)

        self.block_form_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=self.block_form_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Block form fields
        self.create_block_form()

        # Initial population
        self.populate_blocks_list()

    def create_block_form(self):
        """Create the block editing form"""
        frame = self.block_form_frame

        ttk.Label(frame, text="Block Editor", font=('Arial', 14, 'bold')).grid(row=0, column=0, columnspan=2, pady=10)

        row = 1

        # ID
        ttk.Label(frame, text="ID:").grid(row=row, column=0, sticky='w', padx=5, pady=2)
        self.block_id = ttk.Entry(frame, width=40)
        self.block_id.grid(row=row, column=1, sticky='ew', padx=5, pady=2)
        row += 1

        # Name
        ttk.Label(frame, text="Name:").grid(row=row, column=0, sticky='w', padx=5, pady=2)
        self.block_name = ttk.Entry(frame, width=40)
        self.block_name.grid(row=row, column=1, sticky='ew', padx=5, pady=2)
        row += 1

        # Hardness
        ttk.Label(frame, text="Hardness:").grid(row=row, column=0, sticky='w', padx=5, pady=2)
        self.block_hardness = ttk.Entry(frame, width=40)
        self.block_hardness.grid(row=row, column=1, sticky='ew', padx=5, pady=2)
        row += 1

        # Resistance
        ttk.Label(frame, text="Resistance:").grid(row=row, column=0, sticky='w', padx=5, pady=2)
        self.block_resistance = ttk.Entry(frame, width=40)
        self.block_resistance.grid(row=row, column=1, sticky='ew', padx=5, pady=2)
        row += 1

        # Faster Tool
        ttk.Label(frame, text="Faster Tool:").grid(row=row, column=0, sticky='w', padx=5, pady=2)
        self.block_faster_tool = ttk.Combobox(frame, values=["None", "Sword", "Axe", "Pickaxe", "Shovel", "Hoe"], width=37)
        self.block_faster_tool.grid(row=row, column=1, sticky='ew', padx=5, pady=2)
        row += 1

        # Required Tool
        ttk.Label(frame, text="Required Tool:").grid(row=row, column=0, sticky='w', padx=5, pady=2)
        self.block_required_tool = ttk.Combobox(frame, values=["None", "Sword", "Axe", "Pickaxe", "Shovel", "Hoe"], width=37)
        self.block_required_tool.grid(row=row, column=1, sticky='ew', padx=5, pady=2)
        row += 1

        # Minimum Tool Tier
        ttk.Label(frame, text="Min Tool Tier:").grid(row=row, column=0, sticky='w', padx=5, pady=2)
        self.block_min_tier = ttk.Combobox(frame, values=["Hand", "Wood", "Stone", "Iron", "Diamond", "Netherite"], width=37)
        self.block_min_tier.grid(row=row, column=1, sticky='ew', padx=5, pady=2)
        row += 1

        # Transparent
        self.block_transparent = tk.BooleanVar()
        ttk.Checkbutton(frame, text="Transparent", variable=self.block_transparent).grid(row=row, column=0, columnspan=2, sticky='w', padx=5, pady=2)
        row += 1

        # Solid
        self.block_solid = tk.BooleanVar()
        ttk.Checkbutton(frame, text="Solid", variable=self.block_solid).grid(row=row, column=0, columnspan=2, sticky='w', padx=5, pady=2)
        row += 1

        # Tickable
        self.block_tickable = tk.BooleanVar()
        ttk.Checkbutton(frame, text="Tickable", variable=self.block_tickable).grid(row=row, column=0, columnspan=2, sticky='w', padx=5, pady=2)
        row += 1

        # Light Level
        ttk.Label(frame, text="Light Level (0-15):").grid(row=row, column=0, sticky='w', padx=5, pady=2)
        self.block_light = ttk.Entry(frame, width=40)
        self.block_light.grid(row=row, column=1, sticky='ew', padx=5, pady=2)
        row += 1

        # Textures section
        ttk.Label(frame, text="Textures", font=('Arial', 12, 'bold')).grid(row=row, column=0, columnspan=2, pady=10)
        row += 1

        ttk.Label(frame, text="All:").grid(row=row, column=0, sticky='w', padx=5, pady=2)
        self.block_tex_all = ttk.Entry(frame, width=40)
        self.block_tex_all.grid(row=row, column=1, sticky='ew', padx=5, pady=2)
        row += 1

        for side in ["top", "bottom", "north", "south", "east", "west"]:
            ttk.Label(frame, text=f"{side.capitalize()}:").grid(row=row, column=0, sticky='w', padx=5, pady=2)
            entry = ttk.Entry(frame, width=40)
            entry.grid(row=row, column=1, sticky='ew', padx=5, pady=2)
            setattr(self, f"block_tex_{side}", entry)
            row += 1

        # Drops section
        ttk.Label(frame, text="Drops (JSON)", font=('Arial', 12, 'bold')).grid(row=row, column=0, columnspan=2, pady=10)
        row += 1

        self.block_drops = scrolledtext.ScrolledText(frame, width=50, height=6)
        self.block_drops.grid(row=row, column=0, columnspan=2, padx=5, pady=2, sticky='ew')
        row += 1

        # Save button
        ttk.Button(frame, text="Save Block Changes", command=self.save_block_changes).grid(row=row, column=0, columnspan=2, pady=10)

        frame.columnconfigure(1, weight=1)

    def populate_blocks_list(self, filter_text=""):
        """Populate the blocks listbox"""
        self.blocks_list.delete(0, tk.END)
        for block in self.blocks_data.get("blocks", []):
            block_name = f"{block.get('name', 'Unnamed')} ({block.get('id', 'no-id')})"
            if filter_text.lower() in block_name.lower():
                self.blocks_list.insert(tk.END, block_name)

    def filter_blocks(self):
        """Filter blocks based on search"""
        self.populate_blocks_list(self.block_search.get())

    def on_block_select(self, event):
        """Handle block selection"""
        selection = self.blocks_list.curselection()
        if not selection:
            return

        index = selection[0]
        # Find the actual block (accounting for filtering)
        block_name = self.blocks_list.get(index)
        block_id = block_name.split('(')[1].rstrip(')')

        block = None
        for b in self.blocks_data.get("blocks", []):
            if b.get("id") == block_id:
                block = b
                break

        if block:
            self.load_block_into_form(block)

    def load_block_into_form(self, block):
        """Load a block's data into the form"""
        self.block_id.delete(0, tk.END)
        self.block_id.insert(0, block.get("id", ""))

        self.block_name.delete(0, tk.END)
        self.block_name.insert(0, block.get("name", ""))

        self.block_hardness.delete(0, tk.END)
        self.block_hardness.insert(0, str(block.get("hardness", 0)))

        self.block_resistance.delete(0, tk.END)
        self.block_resistance.insert(0, str(block.get("resistance", 0)))

        self.block_faster_tool.set(block.get("fasterTool", "None"))
        self.block_required_tool.set(block.get("requiredTool", "None"))
        self.block_min_tier.set(block.get("minimumToolTier", "Hand"))

        self.block_transparent.set(block.get("transparent", False))
        self.block_solid.set(block.get("solid", True))
        self.block_tickable.set(block.get("tickable", False))

        self.block_light.delete(0, tk.END)
        self.block_light.insert(0, str(block.get("lightLevel", 0)))

        # Textures
        textures = block.get("textures", {})
        self.block_tex_all.delete(0, tk.END)
        self.block_tex_all.insert(0, textures.get("all", ""))

        for side in ["top", "bottom", "north", "south", "east", "west"]:
            entry = getattr(self, f"block_tex_{side}")
            entry.delete(0, tk.END)
            entry.insert(0, textures.get(side, ""))

        # Drops
        self.block_drops.delete('1.0', tk.END)
        self.block_drops.insert('1.0', json.dumps(block.get("drops", []), indent=2))

    def new_block(self):
        """Create a new block"""
        new_block = {
            "id": "minecraft:new_block",
            "name": "New Block",
            "hardness": 1.0,
            "resistance": 1.0,
            "drops": [],
            "textures": {"all": ""},
            "transparent": False,
            "solid": True
        }
        self.blocks_data["blocks"].append(new_block)
        self.populate_blocks_list()
        self.blocks_list.selection_clear(0, tk.END)
        self.blocks_list.selection_set(tk.END)
        self.load_block_into_form(new_block)

    def delete_block(self):
        """Delete the selected block"""
        selection = self.blocks_list.curselection()
        if not selection:
            messagebox.showwarning("Warning", "No block selected")
            return

        if messagebox.askyesno("Confirm", "Delete this block?"):
            block_name = self.blocks_list.get(selection[0])
            block_id = block_name.split('(')[1].rstrip(')')

            self.blocks_data["blocks"] = [b for b in self.blocks_data["blocks"] if b.get("id") != block_id]
            self.populate_blocks_list()

    def save_block_changes(self):
        """Save changes to the current block"""
        try:
            block_id = self.block_id.get()
            if not block_id:
                messagebox.showerror("Error", "Block ID is required")
                return

            # Find or create block
            block = None
            for b in self.blocks_data["blocks"]:
                if b.get("id") == block_id:
                    block = b
                    break

            if not block:
                block = {}
                self.blocks_data["blocks"].append(block)

            # Update block data
            block["id"] = block_id
            block["name"] = self.block_name.get()
            block["hardness"] = float(self.block_hardness.get())
            block["resistance"] = float(self.block_resistance.get())

            if self.block_faster_tool.get() != "None":
                block["fasterTool"] = self.block_faster_tool.get()
            elif "fasterTool" in block:
                del block["fasterTool"]

            if self.block_required_tool.get() != "None":
                block["requiredTool"] = self.block_required_tool.get()
            elif "requiredTool" in block:
                del block["requiredTool"]

            if self.block_min_tier.get() != "Hand":
                block["minimumToolTier"] = self.block_min_tier.get()
            elif "minimumToolTier" in block:
                del block["minimumToolTier"]

            block["transparent"] = self.block_transparent.get()
            block["solid"] = self.block_solid.get()

            if self.block_tickable.get():
                block["tickable"] = True
            elif "tickable" in block:
                del block["tickable"]

            block["lightLevel"] = int(self.block_light.get() or 0)

            # Textures
            textures = {}
            if self.block_tex_all.get():
                textures["all"] = self.block_tex_all.get()
            else:
                for side in ["top", "bottom", "north", "south", "east", "west"]:
                    val = getattr(self, f"block_tex_{side}").get()
                    if val:
                        textures[side] = val
            block["textures"] = textures

            # Drops
            drops_text = self.block_drops.get('1.0', tk.END).strip()
            if drops_text:
                block["drops"] = json.loads(drops_text)
            else:
                block["drops"] = []

            self.populate_blocks_list()
            messagebox.showinfo("Success", "Block updated!")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to save block: {e}")

    def setup_items_tab(self):
        """Setup the items editor tab"""
        # Left panel
        left_frame = ttk.Frame(self.items_tab)
        left_frame.pack(side='left', fill='both', expand=False, padx=5, pady=5)

        ttk.Label(left_frame, text="Items:", font=('Arial', 12, 'bold')).pack()

        # Search
        search_frame = ttk.Frame(left_frame)
        search_frame.pack(fill='x', pady=5)
        ttk.Label(search_frame, text="Search:").pack(side='left')
        self.item_search = ttk.Entry(search_frame)
        self.item_search.pack(side='left', fill='x', expand=True, padx=5)
        self.item_search.bind('<KeyRelease>', lambda e: self.filter_items())

        # Listbox
        list_frame = ttk.Frame(left_frame)
        list_frame.pack(fill='both', expand=True)

        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side='right', fill='y')

        self.items_list = tk.Listbox(list_frame, yscrollcommand=scrollbar.set, width=30)
        self.items_list.pack(side='left', fill='both', expand=True)
        scrollbar.config(command=self.items_list.yview)
        self.items_list.bind('<<ListboxSelect>>', self.on_item_select)

        # Buttons
        btn_frame = ttk.Frame(left_frame)
        btn_frame.pack(fill='x', pady=5)
        ttk.Button(btn_frame, text="New Item", command=self.new_item).pack(fill='x', pady=2)
        ttk.Button(btn_frame, text="Delete Item", command=self.delete_item).pack(fill='x', pady=2)
        ttk.Button(btn_frame, text="Save Items", command=self.save_items).pack(fill='x', pady=2)

        # Right panel
        right_frame = ttk.Frame(self.items_tab)
        right_frame.pack(side='right', fill='both', expand=True, padx=5, pady=5)

        canvas = tk.Canvas(right_frame)
        scrollbar = ttk.Scrollbar(right_frame, orient="vertical", command=canvas.yview)
        self.item_form_frame = ttk.Frame(canvas)

        self.item_form_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=self.item_form_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self.create_item_form()
        self.populate_items_list()

    def create_item_form(self):
        """Create the item editing form"""
        frame = self.item_form_frame

        ttk.Label(frame, text="Item Editor", font=('Arial', 14, 'bold')).grid(row=0, column=0, columnspan=2, pady=10)

        row = 1

        # ID
        ttk.Label(frame, text="ID:").grid(row=row, column=0, sticky='w', padx=5, pady=2)
        self.item_id = ttk.Entry(frame, width=40)
        self.item_id.grid(row=row, column=1, sticky='ew', padx=5, pady=2)
        row += 1

        # Name
        ttk.Label(frame, text="Name:").grid(row=row, column=0, sticky='w', padx=5, pady=2)
        self.item_name = ttk.Entry(frame, width=40)
        self.item_name.grid(row=row, column=1, sticky='ew', padx=5, pady=2)
        row += 1

        # Texture
        ttk.Label(frame, text="Texture:").grid(row=row, column=0, sticky='w', padx=5, pady=2)
        self.item_texture = ttk.Entry(frame, width=40)
        self.item_texture.grid(row=row, column=1, sticky='ew', padx=5, pady=2)
        row += 1

        # Max Stack Size
        ttk.Label(frame, text="Max Stack Size:").grid(row=row, column=0, sticky='w', padx=5, pady=2)
        self.item_max_stack = ttk.Entry(frame, width=40)
        self.item_max_stack.grid(row=row, column=1, sticky='ew', padx=5, pady=2)
        row += 1

        # Kind
        ttk.Label(frame, text="Kind:").grid(row=row, column=0, sticky='w', padx=5, pady=2)
        self.item_kind = ttk.Combobox(frame, values=["block", "tool", "food"], width=37)
        self.item_kind.grid(row=row, column=1, sticky='ew', padx=5, pady=2)
        self.item_kind.bind('<<ComboboxSelected>>', self.on_item_kind_change)
        row += 1

        # Type-specific frame
        self.item_specific_frame = ttk.LabelFrame(frame, text="Type-Specific Properties")
        self.item_specific_frame.grid(row=row, column=0, columnspan=2, sticky='ew', padx=5, pady=10)
        row += 1

        # Save button
        ttk.Button(frame, text="Save Item Changes", command=self.save_item_changes).grid(row=row, column=0, columnspan=2, pady=10)

        frame.columnconfigure(1, weight=1)

    def on_item_kind_change(self, event=None):
        """Update form when item kind changes"""
        # Clear specific frame
        for widget in self.item_specific_frame.winfo_children():
            widget.destroy()

        kind = self.item_kind.get()

        if kind == "block":
            ttk.Label(self.item_specific_frame, text="Block ID:").grid(row=0, column=0, sticky='w', padx=5, pady=2)
            self.item_block_id = ttk.Entry(self.item_specific_frame, width=30)
            self.item_block_id.grid(row=0, column=1, sticky='ew', padx=5, pady=2)

        elif kind == "tool":
            ttk.Label(self.item_specific_frame, text="Tool Type:").grid(row=0, column=0, sticky='w', padx=5, pady=2)
            self.item_tool_type = ttk.Combobox(self.item_specific_frame,
                                               values=["Sword", "Axe", "Pickaxe", "Shovel", "Hoe", "Bow", "Crossbow"], width=27)
            self.item_tool_type.grid(row=0, column=1, sticky='ew', padx=5, pady=2)

            ttk.Label(self.item_specific_frame, text="Tier:").grid(row=1, column=0, sticky='w', padx=5, pady=2)
            self.item_tool_tier = ttk.Combobox(self.item_specific_frame,
                                               values=["Hand", "Wood", "Stone", "Iron", "Diamond", "Netherite"], width=27)
            self.item_tool_tier.grid(row=1, column=1, sticky='ew', padx=5, pady=2)

            ttk.Label(self.item_specific_frame, text="Damage:").grid(row=2, column=0, sticky='w', padx=5, pady=2)
            self.item_tool_damage = ttk.Entry(self.item_specific_frame, width=30)
            self.item_tool_damage.grid(row=2, column=1, sticky='ew', padx=5, pady=2)

            ttk.Label(self.item_specific_frame, text="Durability:").grid(row=3, column=0, sticky='w', padx=5, pady=2)
            self.item_tool_durability = ttk.Entry(self.item_specific_frame, width=30)
            self.item_tool_durability.grid(row=3, column=1, sticky='ew', padx=5, pady=2)

            ttk.Label(self.item_specific_frame, text="Attack Speed:").grid(row=4, column=0, sticky='w', padx=5, pady=2)
            self.item_tool_attack_speed = ttk.Entry(self.item_specific_frame, width=30)
            self.item_tool_attack_speed.grid(row=4, column=1, sticky='ew', padx=5, pady=2)

        elif kind == "food":
            ttk.Label(self.item_specific_frame, text="Hunger:").grid(row=0, column=0, sticky='w', padx=5, pady=2)
            self.item_food_hunger = ttk.Entry(self.item_specific_frame, width=30)
            self.item_food_hunger.grid(row=0, column=1, sticky='ew', padx=5, pady=2)

            ttk.Label(self.item_specific_frame, text="Saturation:").grid(row=1, column=0, sticky='w', padx=5, pady=2)
            self.item_food_saturation = ttk.Entry(self.item_specific_frame, width=30)
            self.item_food_saturation.grid(row=1, column=1, sticky='ew', padx=5, pady=2)

            ttk.Label(self.item_specific_frame, text="Health:").grid(row=2, column=0, sticky='w', padx=5, pady=2)
            self.item_food_health = ttk.Entry(self.item_specific_frame, width=30)
            self.item_food_health.grid(row=2, column=1, sticky='ew', padx=5, pady=2)

    def populate_items_list(self, filter_text=""):
        """Populate the items listbox"""
        self.items_list.delete(0, tk.END)
        for item in self.items_data.get("items", []):
            item_name = f"{item.get('name', 'Unnamed')} ({item.get('id', 'no-id')})"
            if filter_text.lower() in item_name.lower():
                self.items_list.insert(tk.END, item_name)

    def filter_items(self):
        """Filter items based on search"""
        self.populate_items_list(self.item_search.get())

    def on_item_select(self, event):
        """Handle item selection"""
        selection = self.items_list.curselection()
        if not selection:
            return

        item_name = self.items_list.get(selection[0])
        item_id = item_name.split('(')[1].rstrip(')')

        item = None
        for i in self.items_data.get("items", []):
            if i.get("id") == item_id:
                item = i
                break

        if item:
            self.load_item_into_form(item)

    def load_item_into_form(self, item):
        """Load an item's data into the form"""
        self.item_id.delete(0, tk.END)
        self.item_id.insert(0, item.get("id", ""))

        self.item_name.delete(0, tk.END)
        self.item_name.insert(0, item.get("name", ""))

        self.item_texture.delete(0, tk.END)
        self.item_texture.insert(0, item.get("texture", ""))

        self.item_max_stack.delete(0, tk.END)
        self.item_max_stack.insert(0, str(item.get("maxStackSize", 64)))

        self.item_kind.set(item.get("kind", "block"))
        self.on_item_kind_change()

        # Load kind-specific data
        if item.get("kind") == "block":
            if hasattr(self, 'item_block_id'):
                self.item_block_id.delete(0, tk.END)
                self.item_block_id.insert(0, item.get("blockId", ""))
        elif item.get("kind") == "tool":
            if hasattr(self, 'item_tool_type'):
                self.item_tool_type.set(item.get("toolType", ""))
                self.item_tool_tier.set(item.get("tier", ""))
                self.item_tool_damage.delete(0, tk.END)
                self.item_tool_damage.insert(0, str(item.get("damage", 0)))
                self.item_tool_durability.delete(0, tk.END)
                self.item_tool_durability.insert(0, str(item.get("durability", "")))
                self.item_tool_attack_speed.delete(0, tk.END)
                self.item_tool_attack_speed.insert(0, str(item.get("attackSpeed", 1.0)))
        elif item.get("kind") == "food":
            if hasattr(self, 'item_food_hunger'):
                self.item_food_hunger.delete(0, tk.END)
                self.item_food_hunger.insert(0, str(item.get("hunger", 0)))
                self.item_food_saturation.delete(0, tk.END)
                self.item_food_saturation.insert(0, str(item.get("saturation", 0)))
                self.item_food_health.delete(0, tk.END)
                self.item_food_health.insert(0, str(item.get("health", 0)))

    def new_item(self):
        """Create a new item"""
        new_item = {
            "id": "minecraft:new_item",
            "name": "New Item",
            "texture": "",
            "maxStackSize": 64,
            "kind": "block",
            "blockId": ""
        }
        self.items_data["items"].append(new_item)
        self.populate_items_list()
        self.items_list.selection_clear(0, tk.END)
        self.items_list.selection_set(tk.END)
        self.load_item_into_form(new_item)

    def delete_item(self):
        """Delete the selected item"""
        selection = self.items_list.curselection()
        if not selection:
            messagebox.showwarning("Warning", "No item selected")
            return

        if messagebox.askyesno("Confirm", "Delete this item?"):
            item_name = self.items_list.get(selection[0])
            item_id = item_name.split('(')[1].rstrip(')')

            self.items_data["items"] = [i for i in self.items_data["items"] if i.get("id") != item_id]
            self.populate_items_list()

    def save_item_changes(self):
        """Save changes to the current item"""
        try:
            item_id = self.item_id.get()
            if not item_id:
                messagebox.showerror("Error", "Item ID is required")
                return

            # Find or create item
            item = None
            for i in self.items_data["items"]:
                if i.get("id") == item_id:
                    item = i
                    break

            if not item:
                item = {}
                self.items_data["items"].append(item)

            # Update item data
            item["id"] = item_id
            item["name"] = self.item_name.get()
            item["texture"] = self.item_texture.get()
            item["maxStackSize"] = int(self.item_max_stack.get())
            item["kind"] = self.item_kind.get()

            # Kind-specific data
            if item["kind"] == "block" and hasattr(self, 'item_block_id'):
                item["blockId"] = self.item_block_id.get()
            elif item["kind"] == "tool" and hasattr(self, 'item_tool_type'):
                item["toolType"] = self.item_tool_type.get()
                item["tier"] = self.item_tool_tier.get()
                item["damage"] = float(self.item_tool_damage.get())
                durability = self.item_tool_durability.get()
                item["durability"] = int(durability) if durability else None
                item["attackSpeed"] = float(self.item_tool_attack_speed.get())
            elif item["kind"] == "food" and hasattr(self, 'item_food_hunger'):
                item["hunger"] = int(self.item_food_hunger.get())
                item["saturation"] = float(self.item_food_saturation.get())
                item["health"] = int(self.item_food_health.get())

            self.populate_items_list()
            messagebox.showinfo("Success", "Item updated!")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to save item: {e}")

    def setup_crafting_tab(self):
        """Setup the crafting recipes editor tab"""
        # Left panel
        left_frame = ttk.Frame(self.crafting_tab)
        left_frame.pack(side='left', fill='both', expand=False, padx=5, pady=5)

        ttk.Label(left_frame, text="Recipes:", font=('Arial', 12, 'bold')).pack()

        # Search
        search_frame = ttk.Frame(left_frame)
        search_frame.pack(fill='x', pady=5)
        ttk.Label(search_frame, text="Search:").pack(side='left')
        self.recipe_search = ttk.Entry(search_frame)
        self.recipe_search.pack(side='left', fill='x', expand=True, padx=5)
        self.recipe_search.bind('<KeyRelease>', lambda e: self.filter_recipes())

        # Listbox
        list_frame = ttk.Frame(left_frame)
        list_frame.pack(fill='both', expand=True)

        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side='right', fill='y')

        self.recipes_list = tk.Listbox(list_frame, yscrollcommand=scrollbar.set, width=30)
        self.recipes_list.pack(side='left', fill='both', expand=True)
        scrollbar.config(command=self.recipes_list.yview)
        self.recipes_list.bind('<<ListboxSelect>>', self.on_recipe_select)

        # Buttons
        btn_frame = ttk.Frame(left_frame)
        btn_frame.pack(fill='x', pady=5)
        ttk.Button(btn_frame, text="New Recipe", command=self.new_recipe).pack(fill='x', pady=2)
        ttk.Button(btn_frame, text="Delete Recipe", command=self.delete_recipe).pack(fill='x', pady=2)
        ttk.Button(btn_frame, text="Save Recipes", command=self.save_crafting).pack(fill='x', pady=2)

        # Right panel
        right_frame = ttk.Frame(self.crafting_tab)
        right_frame.pack(side='right', fill='both', expand=True, padx=5, pady=5)

        ttk.Label(right_frame, text="Recipe Editor (JSON)", font=('Arial', 14, 'bold')).pack(pady=10)
        ttk.Label(right_frame, text="Edit the recipe as JSON below:").pack()

        self.recipe_json = scrolledtext.ScrolledText(right_frame, width=60, height=30)
        self.recipe_json.pack(fill='both', expand=True, padx=5, pady=5)

        ttk.Button(right_frame, text="Save Recipe Changes", command=self.save_recipe_changes).pack(pady=10)

        self.populate_recipes_list()

    def populate_recipes_list(self, filter_text=""):
        """Populate the recipes listbox"""
        self.recipes_list.delete(0, tk.END)
        for idx, recipe in enumerate(self.crafting_data.get("recipes", [])):
            result = recipe.get("result", {})
            recipe_name = f"{result.get('itemId', 'unknown')} [{recipe.get('kind', 'unknown')}] #{idx}"
            if filter_text.lower() in recipe_name.lower():
                self.recipes_list.insert(tk.END, recipe_name)

    def filter_recipes(self):
        """Filter recipes based on search"""
        self.populate_recipes_list(self.recipe_search.get())

    def on_recipe_select(self, event):
        """Handle recipe selection"""
        selection = self.recipes_list.curselection()
        if not selection:
            return

        recipe_name = self.recipes_list.get(selection[0])
        idx = int(recipe_name.split('#')[1])

        recipe = self.crafting_data["recipes"][idx]
        self.load_recipe_into_form(recipe)

    def load_recipe_into_form(self, recipe):
        """Load a recipe's data into the form"""
        self.recipe_json.delete('1.0', tk.END)
        self.recipe_json.insert('1.0', json.dumps(recipe, indent=2))

    def new_recipe(self):
        """Create a new recipe"""
        new_recipe = {
            "kind": "shapeless",
            "ingredients": [],
            "result": {
                "itemId": "minecraft:new_item",
                "count": 1
            },
            "station": "none"
        }
        self.crafting_data["recipes"].append(new_recipe)
        self.populate_recipes_list()
        self.recipes_list.selection_clear(0, tk.END)
        self.recipes_list.selection_set(tk.END)
        self.load_recipe_into_form(new_recipe)

    def delete_recipe(self):
        """Delete the selected recipe"""
        selection = self.recipes_list.curselection()
        if not selection:
            messagebox.showwarning("Warning", "No recipe selected")
            return

        if messagebox.askyesno("Confirm", "Delete this recipe?"):
            recipe_name = self.recipes_list.get(selection[0])
            idx = int(recipe_name.split('#')[1])

            del self.crafting_data["recipes"][idx]
            self.populate_recipes_list()

    def save_recipe_changes(self):
        """Save changes to the current recipe"""
        try:
            recipe_json = self.recipe_json.get('1.0', tk.END).strip()
            if not recipe_json:
                messagebox.showerror("Error", "Recipe JSON cannot be empty")
                return

            recipe = json.loads(recipe_json)

            # Find which recipe to update based on current selection
            selection = self.recipes_list.curselection()
            if selection:
                recipe_name = self.recipes_list.get(selection[0])
                idx = int(recipe_name.split('#')[1])
                self.crafting_data["recipes"][idx] = recipe
            else:
                # Add as new recipe
                self.crafting_data["recipes"].append(recipe)

            self.populate_recipes_list()
            messagebox.showinfo("Success", "Recipe updated!")

        except json.JSONDecodeError as e:
            messagebox.showerror("Error", f"Invalid JSON: {e}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save recipe: {e}")

def main():
    root = tk.Tk()
    app = GameDataEditor(root)
    root.mainloop()

if __name__ == "__main__":
    main()
