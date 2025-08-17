import flet as ft
from flet.matplotlib_chart import MatplotlibChart
import matplotlib.pyplot as plt
import warnings
import asyncio
from setting_page import create_settings_page  
from components import (
    create_metrics,
    create_metrics_container,
    create_charts,
    create_training_controls,
)
from strategy import load_data
from training import run_training
from model_loader import load_and_test_model

warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

METRIC_KEYS = [
    "Buys", "Sells", "Hold", "Profit", "% Success",
    "Current Money in Wallet", "Assets",
    "% Financial Success", "Average Financial Success",
    "Steps", "Epsilon", "Successful Trades", "Failed Trades"
]

def main(page: ft.Page):
    page.title = "DQN Trading Environment"
    page.padding = 10
    page.window_maximized = True
    page.scroll = "auto"

    page.agent = None

    try:
        df, X, Y = load_data()
        page.df, page.X, page.Y = df, X, Y
    except Exception as e:
        page.add(ft.Text(f"Error loading data: {e}", color=ft.Colors.RED))
        return

    current_period = {"start": "2024-01-01", "end": "2025-01-01"}

    metrics = {key: ft.TextField(label=key, value="0", read_only=True, width=140) for key in METRIC_KEYS}
    metrics_container = ft.GridView(
        expand=True, max_extent=150, child_aspect_ratio=3,
        run_spacing=10, spacing=10, controls=list(metrics.values())
    )

    chart_all, chart_dynamic, chart_pred, ax_all, ax_dynamic, ax_pred = create_charts(df, current_period)

    status, log_text, ai_notes_text, lessons_text, \
    start_btn, pause_btn, end_btn, clear_notes_btn, clear_lessons_btn, \
    mode_dropdown, initial_cash_field, risk_dropdown, select_model_btn, model_name_field = create_training_controls(
        initial_cash=1000,
        update_mode=lambda x: None,
        update_initial_cash=lambda x: None,
        update_risk_level=lambda x: None,
        clear_ai_notes=lambda: None,
        clear_lessons=lambda: None
    )

    count_field = ft.TextField(label="Training Count", value="10", width=120, hint_text="تعداد epoch‌ها")
    speed_field = ft.TextField(label="Training Speed (s/step)", value="0.1", width=120, hint_text="سرعت آموزش")

    model_file_picker = ft.FilePicker()

    training_manager = {
        "update_mode": None,
        "update_initial_cash": None,
        "update_risk_level": None,
        "clear_ai_notes": None,
        "clear_lessons": None,
        "pause_requested": False,
        "training_active": True
    }

    page.controls_dict = {
        "status": status, "log_text": log_text, "ai_notes_text": ai_notes_text, "lessons_text": lessons_text,
        "start_btn": start_btn, "pause_btn": pause_btn, "end_btn": end_btn,
        "clear_notes_btn": clear_notes_btn, "clear_lessons_btn": clear_lessons_btn,
        "mode_dropdown": mode_dropdown, "initial_cash_field": initial_cash_field,
        "risk_dropdown": risk_dropdown, "select_model_btn": select_model_btn,
        "model_name_field": model_name_field, "count_field": count_field, "speed_field": speed_field,
        "metrics": metrics, "chart_all": chart_all, "chart_dynamic": chart_dynamic,
        "chart_pred": chart_pred, "ax_all": ax_all, "ax_dynamic": ax_dynamic, "ax_pred": ax_pred
    }

    page.training_manager = training_manager

    def open_settings(e):
        page.views.append(create_settings_page(page))
        page.update()

    settings_btn = ft.ElevatedButton("Learning Settings", width=150, on_click=open_settings)

    def handle_model_selection(e):
        if e.files:
            model_path = e.files[0].path
            model_name_field.value = model_path
            model_name_field.update()
            status.value = f"Testing model: {model_path}"
            status.color = ft.Colors.BLUE
            status.update()
            load_and_test_model(
                model_path, log_text, metrics,
                chart_all, chart_dynamic, chart_pred,
                ax_all, ax_dynamic, ax_pred,
                lessons_text, ai_notes_text,
                df, X, Y,
                initial_cash_field.value, risk_dropdown.value
            )
        else:
            model_name_field.value = ""
            model_name_field.update()
            status.value = "No model selected."
            status.color = ft.Colors.RED
            status.update()

    def update_mode(mode):
        log_text.value += f"Mode updated to {mode}\n"
        log_text.update()

    def update_initial_cash(cash):
        try:
            cash = float(cash)
            log_text.value += f"Initial portfolio updated to {cash}\n"
        except ValueError:
            log_text.value += "Invalid portfolio value entered.\n"
        log_text.update()

    def update_risk_level(risk):
        log_text.value += f"Risk level updated to {risk}\n"
        log_text.update()

    def clear_ai_notes():
        ai_notes_text.value = ""
        ai_notes_text.update()

    def clear_lessons():
        lessons_text.value = ""
        lessons_text.update()

    def start_click(e):
        start_btn.disabled = True
        pause_btn.disabled = False
        end_btn.disabled = False
        select_model_btn.disabled = True
        training_manager["training_active"] = True

        try:
            count = int(count_field.value)
        except ValueError:
            count = 10
        try:
            delay = float(speed_field.value)
        except ValueError:
            delay = 0.1

        training_manager.update({
            "update_mode": update_mode,
            "update_initial_cash": update_initial_cash,
            "update_risk_level": update_risk_level,
            "clear_ai_notes": clear_ai_notes,
            "clear_lessons": clear_lessons
        })

        mode_dropdown.on_change = lambda e: update_mode(e.control.value)
        initial_cash_field.on_change = lambda e: update_initial_cash(e.control.value)
        risk_dropdown.on_change = lambda e: update_risk_level(e.control.value)
        clear_notes_btn.on_click = lambda e: clear_ai_notes()
        clear_lessons_btn.on_click = lambda e: clear_lessons()

        page.run_task(
            run_training,
            page, status, log_text, ai_notes_text, lessons_text,
            metrics, chart_all, chart_dynamic, chart_pred,
            ax_all, ax_dynamic, ax_pred,
            count, delay, training_manager,
            df, X, Y, initial_cash_field.value, risk_dropdown.value,
            set_agent=lambda agent: setattr(page, 'agent', agent)
        )

    def pause_click(e):
        training_manager["pause_requested"] = not training_manager["pause_requested"]
        status.value = "Training Paused" if training_manager["pause_requested"] else "Training Resumed"
        status.color = ft.Colors.YELLOW if training_manager["pause_requested"] else ft.Colors.BLUE
        status.update()

    def end_click(e):
        training_manager["training_active"] = False
        start_btn.disabled = False
        pause_btn.disabled = True
        end_btn.disabled = True
        select_model_btn.disabled = False
        status.value = "Training Stopped."
        status.color = ft.Colors.RED
        status.update()

    def select_model_click(e):
        page.overlay.append(model_file_picker)
        model_file_picker.on_result = handle_model_selection
        model_file_picker.pick_files()
        page.update()

    def on_window_close(e):
        if e.data == "close" and page.agent is not None:
            try:
                page.agent.save_model("dqn_model_auto_save.pt")
                log_text.value += "Model auto-saved as 'dqn_model_auto_save.pt' on window close.\n"
                log_text.update()
            except Exception as ex:
                log_text.value += f"Error auto-saving model: {ex}\n"
                log_text.update()
        page.close()

    start_btn.on_click = start_click
    pause_btn.on_click = pause_click
    end_btn.on_click = end_click
    select_model_btn.on_click = select_model_click
    page.on_window_event = on_window_close

    page.add(
        ft.Column([
            ft.Row([settings_btn]),
            ft.Row([
                ft.Container(chart_all, expand=True),
                ft.Container(chart_dynamic, expand=True),
                ft.Container(chart_pred, expand=True),
            ], expand=True),
            ft.Row([
                ft.Column([
                    status,
                    ft.Row([start_btn, pause_btn, end_btn, clear_notes_btn, clear_lessons_btn, select_model_btn, count_field, speed_field]),
                    ft.Row([mode_dropdown, initial_cash_field, risk_dropdown, model_name_field]),
                    log_text
                ], expand=2),
                metrics_container
            ], expand=True),
            ft.Row([ft.Text("Lessons & Notes:", weight="bold")]),
            ft.Row([
                ft.Container(lessons_text, expand=True),
                ft.Container(ai_notes_text, expand=True)
            ], expand=True)
        ], scroll="auto", expand=True)
    )

    page.overlay.append(model_file_picker)
    page.update()

if __name__ == "__main__":
    ft.app(target=main, view=ft.AppView.FLET_APP)