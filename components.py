import flet as ft
from flet.matplotlib_chart import MatplotlibChart
import matplotlib.pyplot as plt
import matplotlib
import asyncio

matplotlib.use('Agg')

def create_metrics():
    return {
        "Buys": ft.TextField(label="Buys", read_only=True, hint_text="تعداد خرید", width=150),
        "Sells": ft.TextField(label="Sells", read_only=True, hint_text="تعداد فروش", width=150),
        "Hold": ft.TextField(label="Hold", read_only=True, hint_text="تعداد نگهداری", width=150),
        "Profit": ft.TextField(label="Profit", read_only=True, hint_text="سود خالص", width=150),
        "% Success": ft.TextField(label="% Success", read_only=True, hint_text="درصد موفقیت", width=150),
        "Current Money in Wallet": ft.TextField(label="Money in Wallet", read_only=True, hint_text="موجودی", width=150),
        "Assets": ft.TextField(label="Assets", read_only=True, hint_text="دارایی‌ها (واحد)", width=150),
        "% Financial Success": ft.TextField(label="% Financial Success", read_only=True, hint_text="موفقیت مالی", width=150),
        "Average Financial Success": ft.TextField(label="Avg Financial Success", read_only=True, hint_text="میانگین موفقیت", width=150),
        "Steps": ft.TextField(label="Steps", read_only=True, hint_text="تعداد گام‌ها", width=150),
        "Epsilon": ft.TextField(label="Epsilon", read_only=True, hint_text="درجه اکتشاف", width=150),
        "Successful Trades": ft.TextField(label="Successful Trades", read_only=True, hint_text="معاملات موفق", width=150),
        "Failed Trades": ft.TextField(label="Failed Trades", read_only=True, hint_text="معاملات ناموفق", width=150)
    }

def create_metrics_container(metrics):
    return ft.Card(
        content=ft.Container(
            content=ft.Column([
                ft.ResponsiveRow([metrics["Buys"], metrics["Sells"]]),
                ft.ResponsiveRow([metrics["Hold"], metrics["Profit"]]),
                ft.ResponsiveRow([metrics["% Success"]]),
                ft.ResponsiveRow([metrics["Current Money in Wallet"], metrics["Assets"]]),
                ft.ResponsiveRow([metrics["% Financial Success"], metrics["Average Financial Success"]]),
                ft.ResponsiveRow([metrics["Steps"], metrics["Epsilon"]]),
                ft.ResponsiveRow([metrics["Successful Trades"], metrics["Failed Trades"]])
            ], scroll="auto"),
            padding=10
        ),
        elevation=5,
        width=320
    )

def create_charts(df, current_period):
    fig_all, ax_all = plt.subplots()
    fig_dynamic, ax_dynamic = plt.subplots()
    fig_pred, ax_pred = plt.subplots()

    # Chart All: Actual price full timeline
    ax_all.plot(df.index, df["priceClose"], color="gray", label="Actual Price")
    ax_all.set_title(f"Actual Price ({current_period['start'][:4]}-{current_period['end'][:4]})")
    ax_all.legend()

    # Chart Dynamic: placeholder
    ax_dynamic.set_title("Dynamic Predicted vs Actual Price (Last Window)")
    ax_dynamic.set_xlabel("Steps")
    ax_dynamic.set_ylabel("Price")
    ax_dynamic.grid(True)

    # Chart Pred: Actual full + overlay predictions
    ax_pred.plot(df.index, df["priceClose"], color="gray", label="Actual Price")
    ax_pred.set_title("Predicted Price (Full Timeline)")
    ax_pred.legend()

    for fig in (fig_all, fig_dynamic, fig_pred):
        fig.tight_layout()

    return (
        MatplotlibChart(fig_all, expand=True),
        MatplotlibChart(fig_dynamic, expand=True),
        MatplotlibChart(fig_pred, expand=True),
        ax_all, ax_dynamic, ax_pred
    )

def create_training_controls(initial_cash, update_mode, update_initial_cash, update_risk_level, clear_ai_notes, clear_lessons):
    status = ft.Text("Press Start to begin training.", size=16, weight="bold", color=ft.Colors.BLUE)
    log_text = ft.TextField(
        multiline=True,
        read_only=True,
        expand=True,
        min_lines=14,
        max_lines=14,
        label="Training Log",
        hint_text="Log messages will appear here"
    )
    ai_notes_text = ft.TextField(
        multiline=True,
        read_only=True,
        expand=True,
        min_lines=6,
        max_lines=6,
        label="AI Notes",
        hint_text="AI notes will appear here"
    )
    lessons_text = ft.TextField(
        multiline=True,
        read_only=True,
        expand=True,
        min_lines=6,
        max_lines=6,
        label="Lessons Learned",
        hint_text="Lessons learned will appear here"
    )
    
    start_btn = ft.ElevatedButton("Start", width=100)
    pause_btn = ft.ElevatedButton("Pause", disabled=True, width=100)
    end_btn = ft.ElevatedButton("Stop & Save", disabled=True, width=100)
    clear_notes_btn = ft.ElevatedButton("Clear AI Notes", width=100)
    clear_lessons_btn = ft.ElevatedButton("Clear Lessons", width=100)
    select_model_btn = ft.ElevatedButton("Select Model", width=100)
    
    mode_dropdown = ft.Dropdown(
        label="Trading Mode",
        options=[
            ft.dropdown.Option("Reward-Based"),
            ft.dropdown.Option("Money Management")
        ],
        value="Money Management",
        width=200,
        on_change=lambda e: update_mode(e.control.value)
    )
    
    initial_cash_field = ft.TextField(
        label="Initial Portfolio ($)",
        value=str(initial_cash),
        disabled=False,
        hint_text="Enter initial portfolio amount",
        width=200,
        on_change=lambda e: update_initial_cash(e.control.value)
    )
    
    risk_dropdown = ft.Dropdown(
        label="Risk Level",
        options=[
            ft.dropdown.Option("Very High Risk"),
            ft.dropdown.Option("High Risk"),
            ft.dropdown.Option("Medium Risk"),
            ft.dropdown.Option("Low Risk"),
            ft.dropdown.Option("Very Low Risk")
        ],
        value="Medium Risk",
        width=200,
        on_change=lambda e: update_risk_level(e.control.value),
        disabled=False
    )
    
    model_name_field = ft.TextField(
        label="Selected Model",
        read_only=True,
        hint_text="Model path will appear here",
        width=300
    )
    
    return status, log_text, ai_notes_text, lessons_text, start_btn, pause_btn, end_btn, clear_notes_btn, clear_lessons_btn, mode_dropdown, initial_cash_field, risk_dropdown, select_model_btn, model_name_field

def highlight_metric(field: ft.TextField, color: ft.Colors, duration: float = 1.0):
    field.border_color = color
    field.border_width = 2
    field.update()

    async def clear_border():
        # fade out after duration
        await asyncio.sleep(duration)
        field.border_color = ft.Colors.TRANSPARENT
        field.border_width = 1
        field.update()

    asyncio.create_task(clear_border())