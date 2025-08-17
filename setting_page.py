import flet as ft
import json
import os

def create_settings_page(page):
    settings = {
        "gamma": 0.95,
        "epsilon": 1.0,
        "epsilon_min": 0.05,
        "epsilon_decay": 0.995,
        "learning_rate": 0.001,
        "batch_size": 32,
        "forecast_steps": 1,
        "strategy_description": "Predict price movement (±1%) based on current step."
    }

    settings_file = "learning_settings.json"
    if os.path.exists(settings_file):
        try:
            with open(settings_file, "r") as f:
                settings.update(json.load(f))
            page.controls_dict["log_text"].value += "Loaded existing settings from learning_settings.json\n"
        except Exception as e:
            page.controls_dict["log_text"].value += f"Error loading settings: {e}\n"

    gamma_field = ft.TextField(label="Gamma (Discount Factor)", value=str(settings["gamma"]), width=200, hint_text="0.0 to 1.0")
    epsilon_field = ft.TextField(label="Epsilon (Exploration Rate)", value=str(settings["epsilon"]), width=200, hint_text="0.0 to 1.0")
    epsilon_min_field = ft.TextField(label="Epsilon Min", value=str(settings["epsilon_min"]), width=200, hint_text="Minimum exploration")
    epsilon_decay_field = ft.TextField(label="Epsilon Decay", value=str(settings["epsilon_decay"]), width=200, hint_text="Decay rate per episode")
    learning_rate_field = ft.TextField(label="Learning Rate", value=str(settings["learning_rate"]), width=200, hint_text="e.g., 0.001")
    batch_size_field = ft.TextField(label="Batch Size", value=str(settings["batch_size"]), width=200, hint_text="e.g., 32")
    forecast_steps_field = ft.TextField(label="Forecast Steps", value=str(settings["forecast_steps"]), width=200, hint_text="Steps to predict ahead")
    strategy_field = ft.TextField(
        label="Strategy Description",
        value=settings["strategy_description"],
        multiline=True,
        min_lines=3,
        max_lines=5,
        width=400,
        hint_text="Describe the trading strategy"
    )

    def save_settings(e):
        try:
            new_settings = {
                "gamma": float(gamma_field.value) if gamma_field.value.strip() else settings["gamma"],
                "epsilon": float(epsilon_field.value) if epsilon_field.value.strip() else settings["epsilon"],
                "epsilon_min": float(epsilon_min_field.value) if epsilon_min_field.value.strip() else settings["epsilon_min"],
                "epsilon_decay": float(epsilon_decay_field.value) if epsilon_decay_field.value.strip() else settings["epsilon_decay"],
                "learning_rate": float(learning_rate_field.value) if learning_rate_field.value.strip() else settings["learning_rate"],
                "batch_size": int(batch_size_field.value) if batch_size_field.value.strip() else settings["batch_size"],
                "forecast_steps": int(forecast_steps_field.value) if forecast_steps_field.value.strip() else settings["forecast_steps"],
                "strategy_description": strategy_field.value or settings["strategy_description"]
            }
            with open(settings_file, "w") as f:
                json.dump(new_settings, f, indent=4)
            page.controls_dict["log_text"].value += "Settings saved to learning_settings.json\n"
            page.views.pop()
            page.controls_dict["log_text"].value += "Returned to main page.\n"
            page.update()
        except ValueError as ve:
            page.controls_dict["log_text"].value += f"Invalid input: {ve}\n"
            page.update()
        except Exception as e:
            page.controls_dict["log_text"].value += f"Error saving settings: {e}\n"
            page.update()

    def cancel_settings(e):
        try:
            page.controls_dict["log_text"].value += "Settings page cancelled.\n"
            page.views.pop()
            page.controls_dict["log_text"].value += "Returned to main page.\n"
            page.update()
        except Exception as e:
            page.controls_dict["log_text"].value += f"Error cancelling settings: {e}\n"
            page.update()

    save_btn = ft.ElevatedButton("Save/Use", width=100, on_click=save_settings)
    cancel_btn = ft.ElevatedButton("Cancel", width=100, on_click=cancel_settings)
    back_btn = ft.ElevatedButton("<", width=150, on_click=cancel_settings)
    return ft.View(
        route="/settings",
        controls=[
            ft.AppBar(title=ft.Text("Learning Settings"),leading=back_btn, bgcolor=ft.Colors.BLUE_200,),
            ft.Column([
                ft.Row([gamma_field, epsilon_field]),
                ft.Row([epsilon_min_field, epsilon_decay_field]),
                ft.Row([learning_rate_field, batch_size_field]),
                ft.Row([forecast_steps_field]),
                ft.Row([strategy_field]),
                ft.Row([save_btn, cancel_btn]),
            ], scroll="auto", expand=True),
        ],
        padding=20
    )