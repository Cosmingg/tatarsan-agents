import base64
import csv
import datetime
import io
import json
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from processing_graph import (
    build_processing_graph,
    RequestFields,
    ALL_TU_CONFIGS,
    DEFAULT_TU_ID,
)

# Компилируем граф один раз при старте
GRAPH = build_processing_graph()


# ---------- Утилиты ----------

def ensure_dirs():
    Path("uploads").mkdir(exist_ok=True)


def save_uploaded_file(uploaded, dest_dir: str = "uploads") -> str:
    ensure_dirs()
    suffix = Path(uploaded.name).suffix
    name = f"{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}{suffix}"
    path = Path(dest_dir) / name
    with open(path, "wb") as f:
        f.write(uploaded.getbuffer())
    return str(path)


def save_submission(image_path: str, fields: List[str], csv_path: str = "submissions.csv") -> None:
    # fields: 10 параметров НЭМС + примечания (11-е поле)
    header = ["timestamp", "image"] + [f"field_{i+1}" for i in range(11)] + ["joined"]
    exists = Path(csv_path).exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(header)
        row = [
            datetime.datetime.now(datetime.timezone.utc).isoformat(),
            image_path,
        ] + fields + [" - ".join(fields)]
        writer.writerow(row)


def read_submissions(csv_path: str = "submissions.csv") -> List[Dict[str, Any]]:
    if not Path(csv_path).exists():
        return []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def request_fields_to_ui_list(fields_dict: dict) -> List[str]:
    """
    Маппим RequestFields (НЭМС) → 10 строк для полей UI.
    Примечания (notes) больше не отдельное поле — они пойдут в блок «Скопировать и сохранить».
    """
    rf = RequestFields(**fields_dict) if fields_dict else RequestFields()
    res: List[str] = [
        str(rf.dn_mm or ""),              # 1 Наружный диаметр патрубков Дн, мм
        str(rf.pressure_kgf_cm2 or ""),   # 2 Рабочее давление, кгс/см²
        str(rf.length_mm or ""),          # 3 Длина изделия, мм
        rf.medium_code or "",             # 4 Среда (код/описание)
        rf.placement_code or "",          # 5 Место размещения (код)
        rf.connection_code or "",         # 6 Соединение (код/описание)
        rf.inner_coating_code or "",      # 7 Внутреннее покрытие (код)
        rf.outer_coating_code or "",      # 8 Наружное покрытие (код)
        rf.terminals_code or "",          # 9 Клеммы (К/пусто)
        rf.climate_code or "",            # 10 Климатическое исполнение
    ]
    return res


def _mapping_to_str(d: Dict[str, Any], tu_id: Optional[str] = None) -> str:
    """
    Красивый вывод словаря код → значение/описание.
    В КАЖДОМ коде добавляем ' ТУ <id>', если tu_id задан.
    """
    suffix = f" ТУ {tu_id}" if tu_id else ""
    parts: List[str] = []
    for k, v in d.items():
        if isinstance(v, dict):
            # Давление
            if "pn_kgf_cm2" in v:
                parts.append(f"{k}{suffix}: {v.get('pn_kgf_cm2')} кгс/см²")
            # Климат
            elif {"climate", "t_oper_min_c", "t_oper_max_c"} <= v.keys():
                parts.append(
                    f"{k}{suffix}: {v['climate']}, tраб {v['t_oper_min_c']}…{v['t_oper_max_c']} °C"
                )
            else:
                parts.append(f"{k}{suffix}: {v}")
        else:
            parts.append(f"{k}{suffix}: {v}")
    return "; ".join(parts)


def build_field_helps(tu_data: Dict[str, Any], tu_id: Optional[str]) -> List[str]:
    """
    Строим подсказки (help) для 10 параметров на основании ТОЛЬКО tu.json.

    Используем:
    - designation_schemes.full.fields_description — текст описаний полей
    - standard_lengths_by_dn, pressure_classes, product_types, placement_types,
      construction_types, internal_coating_types, external_coating_types,
      climate_types, terminal_markings — как словари значений.

    Никаких хардкодов по смыслу полей — всё из JSON.
    """
    designation_schemes = tu_data.get("designation_schemes", {})
    full_scheme = designation_schemes.get("full", {})
    fields_desc: Dict[str, str] = full_scheme.get("fields_description", {})

    standard_lengths_by_dn = tu_data.get("standard_lengths_by_dn", {})
    pressure_classes = tu_data.get("pressure_classes", {})
    product_types = tu_data.get("product_types", {})
    placement_types = tu_data.get("placement_types", {})
    construction_types = tu_data.get("construction_types", {})
    internal_coating_types = tu_data.get("internal_coating_types", {})
    external_coating_types = tu_data.get("external_coating_types", {})
    climate_types = tu_data.get("climate_types", {})
    terminal_markings = tu_data.get("terminal_markings", {})

    # Соответствие: индекс поля UI → ключ в designation_schemes.full.fields_description
    ui_to_json_field = [
        "outer_diameter_mm",      # 0
        "pressure_kgf_cm2",       # 1
        "length_mm",              # 2
        "product_code",           # 3
        "placement_code",         # 4
        "construction_code",      # 5
        "inner_lining_code",      # 6
        "external_covering_code", # 7
        "terminal_mark",          # 8
        "climate_code",           # 9
    ]

    helps: List[str] = []

    for json_field in ui_to_json_field:
        lines: List[str] = []

        # 1) Базовое описание поля из JSON (fields_description)
        base_desc = fields_desc.get(json_field)
        if base_desc:
            lines.append(base_desc)
        else:
            lines.append(f"Поле '{json_field}' согласно ТУ {tu_id}.")

        # 2) Дополнительная информация из соответствующих секций JSON
        if json_field == "outer_diameter_mm" and standard_lengths_by_dn:
            # Доступные диаметры/длины — всё из JSON
            std_pairs = ", ".join(f"{dn}→{L} мм" for dn, L in standard_lengths_by_dn.items())
            lines.append(
                f"standard_lengths_by_dn ТУ {tu_id}: {std_pairs}"
            )

        if json_field == "pressure_kgf_cm2" and pressure_classes:
            lines.append(
                f"pressure_classes ТУ {tu_id}: " + _mapping_to_str(pressure_classes, tu_id)
            )

        if json_field == "product_code" and product_types:
            lines.append(
                f"product_types ТУ {tu_id}: " + _mapping_to_str(product_types, tu_id)
            )

        if json_field == "placement_code" and placement_types:
            lines.append(
                f"placement_types ТУ {tu_id}: " + _mapping_to_str(placement_types, tu_id)
            )

        if json_field == "construction_code" and construction_types:
            lines.append(
                f"construction_types ТУ {tu_id}: " + _mapping_to_str(construction_types, tu_id)
            )

        if json_field == "inner_lining_code" and internal_coating_types:
            lines.append(
                f"internal_coating_types ТУ {tu_id}: " + _mapping_to_str(internal_coating_types, tu_id)
            )

        if json_field == "external_covering_code" and external_coating_types:
            lines.append(
                f"external_coating_types ТУ {tu_id}: " + _mapping_to_str(external_coating_types, tu_id)
            )

        if json_field == "terminal_mark" and terminal_markings:
            lines.append(
                f"terminal_markings ТУ {tu_id}: " + _mapping_to_str(terminal_markings, tu_id)
            )

        if json_field == "climate_code" and climate_types:
            lines.append(
                f"climate_types ТУ {tu_id}: " + _mapping_to_str(climate_types, tu_id)
            )

        helps.append("\n".join(lines))

    return helps


# ---------- Основное приложение ----------

def main() -> None:
    logo_path = Path(__file__).with_name("logo.jpg")

    st.set_page_config(
        page_title="Заявки клиентов",
        page_icon=str(logo_path), 
        layout="wide",
    )
    
    # Лёгкий CSS-твик
    st.markdown(
        """
        <style>
        :root {
          --primary: #027fa9;
          --primary-dark: #026a86;
          --primary-light: #79cde0;
          --text-dark: #083a45;
        }

        .stApp {
            background: linear-gradient(180deg, #f0fbff 0%, #ffffff 100%);
            color: var(--text-dark);
            font-family: 'Helvetica Neue', Arial, sans-serif;
        }

        .stButton>button, .stDownloadButton>button {
            background-color: #027fa9 !important;
            color: white !important;
            border-radius: 6px;
            border: none;
            padding: 8px 12px;
        }
        .stButton>button:hover, .stDownloadButton>button:hover {
            background-color: #026a86 !important;
        }

        .stFileUploader, .stFileUploader div {
            border-radius: 8px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Загрузка заявок и полуавтоматическая обработка (AI)")

    tab_upload, tab_dashboard = st.tabs(["Форма загрузки", "Дашборд"])

    # ------------- Вкладка: Форма загрузки -------------

    with tab_upload:
        # Выводим в консоль браузера сообщения OCR, если они пришли из последнего запуска
        ocr_logs = st.session_state.pop("ocr_console_logs", None)
        if ocr_logs:
            for log in ocr_logs:
                components.html(
                    f"<script>console.log({json.dumps(log)});</script>",
                    height=0,
                )

        # --- выбор технических условий (ТУ) ---
        selected_tu_id: Optional[str] = None
        tu_data_for_help: Optional[Dict[str, Any]] = None

        if ALL_TU_CONFIGS:
            tu_names = {
                cfg["meta"].get("name", tu_id): tu_id
                for tu_id, cfg in ALL_TU_CONFIGS.items()
            }
            tu_name_list = list(tu_names.keys())

            # выбираем дефолт по DEFAULT_TU_ID
            try:
                default_name = next(
                    name for name, tid in tu_names.items() if tid == DEFAULT_TU_ID
                )
                default_index = tu_name_list.index(default_name)
            except StopIteration:
                default_index = 0

            selected_name = st.selectbox(
                "Технические условия (ТУ):",
                options=tu_name_list,
                index=default_index,
            )
            selected_tu_id = tu_names[selected_name]
            st.session_state["selected_tu_id"] = selected_tu_id
            tu_data_for_help = ALL_TU_CONFIGS[selected_tu_id]["data"]
        else:
            st.warning("Не найдены файлы ТУ в папке tu/. Обработка будет без привязки к ТУ.")
            st.session_state["selected_tu_id"] = None

        # Подсказки для 10 полей — целиком из tu.json
        field_helps: Optional[List[str]] = None
        if tu_data_for_help is not None:
            field_helps = build_field_helps(tu_data_for_help, selected_tu_id)

        st.header("1️⃣ Загрузите файл заявки")
        uploaded = st.file_uploader(
            "Выберите файл (png/jpg/jpeg/pdf/docx/xlsx)",
            type=["png", "jpg", "jpeg", "pdf", "docx", "xlsx"],
        )

        saved_path: Optional[str] = None
        if uploaded is not None:
            try:
                saved_path = save_uploaded_file(uploaded)
                st.session_state["last_uploaded_path"] = saved_path
                suffix = Path(saved_path).suffix.lower()

                # Предпросмотр
                if suffix in [".png", ".jpg", ".jpeg"]:
                    st.image(saved_path, use_container_width=True)
                elif suffix == ".pdf":
                    try:
                        with open(saved_path, "rb") as f:
                            pdf_bytes = f.read()
                        b64 = base64.b64encode(pdf_bytes).decode("utf-8")
                        pdf_display = (
                            f"<iframe src='data:application/pdf;base64,{b64}' "
                            "width='100%' height='600' style='border: none;'></iframe>"
                        )
                        components.html(pdf_display, height=600)
                    except Exception:
                        st.info(f"Загружен PDF: {Path(saved_path).name}")
                elif suffix == ".docx":
                    try:
                        from docx import Document

                        doc = Document(saved_path)
                        text = "\n".join([p.text for p in doc.paragraphs if p.text])
                        if text:
                            st.text_area(
                                "Предпросмотр DOCX (первые 2000 символов):",
                                value=text[:2000],
                                height=300,
                            )
                        else:
                            st.info(f"Загружен DOCX: {Path(saved_path).name}")
                    except Exception:
                        st.info(f"Загружен DOCX: {Path(saved_path).name}")
                elif suffix in [".xls", ".xlsx"]:
                    try:
                        df_preview = pd.read_excel(saved_path, nrows=20)
                        st.dataframe(df_preview, use_container_width=True)
                    except Exception:
                        st.info(f"Загружен Excel: {Path(saved_path).name}")
                else:
                    st.info(f"Загружен файл: {Path(saved_path).name}")
                    with open(saved_path, "rb") as f:
                        st.download_button(
                            "Скачать загруженный файл",
                            data=f,
                            file_name=Path(saved_path).name,
                        )

                st.success(f"Файл сохранён: {saved_path}")
            except Exception as e:
                st.error(f"Не удалось сохранить файл: {e}")

        # --- Кнопка AI-сборки сразу после загрузки файла ---
        st.write("---")
        st.subheader("2️⃣ Заполнить поля с помощью AI (LangGraph)")

        if "processing_error" in st.session_state:
            st.error(st.session_state["processing_error"])
            st.session_state.pop("processing_error", None)

        def _fill_from_ai() -> None:
            path = st.session_state.get("last_uploaded_path")
            if not path:
                st.session_state["processing_error"] = "Сначала загрузите файл заявки."
                return

            try:
                state: Dict[str, Any] = {
                    "file_path": path,
                    "messages": [],
                }

                tu_id = st.session_state.get("selected_tu_id")
                if tu_id:
                    state["tu_id"] = tu_id

                result = GRAPH.invoke(state)
                # не сохраняем сырые байты в сессию
                result.pop("file_bytes", None)

                ocr_messages = [
                    m for m in result.get("messages", []) if m.startswith("[ocr]")
                ]
                if ocr_messages:
                    st.session_state["ocr_console_logs"] = ocr_messages
                else:
                    st.session_state.pop("ocr_console_logs", None)

                # ЛОГИ ТОЛЬКО В КОНСОЛЬ (локально), НЕ В UI
                print("\n=== GRAPH RESULT (LOCAL LOG) ===")
                print(json.dumps(result.get("request_fields", {}), ensure_ascii=False, indent=2))
                print("\n--- MESSAGES ---")
                for m in result.get("messages", []):
                    print(m)
                print("=== END GRAPH RESULT ===\n")

                # заполняем поля UI
                rf_dict = result.get("request_fields") or {}
                rf = RequestFields(**rf_dict)

                # 10 параметров НЭМС
                ui_vals = request_fields_to_ui_list(rf_dict)
                for i in range(10):
                    st.session_state[f"field_{i+1}"] = ui_vals[i]

                # примечания в отдельное поле в блоке "Скопировать и сохранить"
                if rf.notes:
                    st.session_state["extra_notes"] = rf.notes
            except Exception as e:
                st.session_state["processing_error"] = f"Ошибка при обработке: {e}"

        st.button(
            "Заполнить поля из файла (AI)",
            on_click=_fill_from_ai,
            disabled=("last_uploaded_path" not in st.session_state),
        )

        # --- Параметры НЭМС после кнопки AI ---
        st.write("---")
        st.subheader("3️⃣ Параметры НЭМС (10 полей)")

        tu_caption_id = selected_tu_id or "3667-013-05608841-2020"
        st.info(
            f"НЭМС — неразъемное электроизолирующее муфтовое соединение "
            f"(см. пример обозначения: НЭМС-325-40-800-ВД-1-2-4-3-К-УД ТУ {tu_caption_id})."
        )

        st.write(
            "AI будет пытаться заполнить поля автоматически из файла, "
            "но вы можете отредактировать их вручную. "
            "Для подсказки наведите курсор на значок вопроса рядом с полем."
        )

        labels = [
            "Наружный диаметр патрубков Дн, мм",                          # 0
            "Рабочее давление, кгс/см²",                                   # 1
            "Длина изделия, мм",                                           # 2
            "Среда (например, техническая или питьевая вода — ВД)",        # 3
            "Место размещения на трубопроводе",                            # 4
            "Соединение с трубопроводом (тип, напр. сварка с наконечником)",  # 5
            "Внутреннее покрытие (код по ТУ)",                             # 6
            "Наружное покрытие (код по ТУ)",                               # 7
            "Установлены клеммы (К / без клемм)",                          # 8
            "Климатическое исполнение по ГОСТ 15150 (код по ТУ)",          # 9
        ]

        fields: List[str] = []
        for i, label in enumerate(labels):
            key = f"field_{i+1}"
            st.session_state.setdefault(key, "")
            help_text = None
            if field_helps and i < len(field_helps):
                help_text = field_helps[i]
            fields.append(
                st.text_input(
                    label,
                    key=key,
                    placeholder="(пусто)",
                    help=help_text,
                )
            )

        # --- Финальная аббревиатура ---
        st.write("**Финальное обозначение НЭМС:**")
        param_values = [st.session_state.get(f"field_{i+1}", "") for i in range(10)]
        non_empty_params = [v for v in param_values if v]
        if non_empty_params:
            designation_core = " - ".join(non_empty_params)
            tu_part = f" ТУ {tu_caption_id}" if tu_caption_id else ""
            final_designation = f"НЭМС - {designation_core}{tu_part}"
        else:
            final_designation = f"НЭМС - … ТУ {tu_caption_id}"
        st.code(final_designation)

        # --- Сохранение заявки + примечания ---
        st.write("---")
        st.subheader("4️⃣ Скопировать и сохранить")

        notes_key = "extra_notes"
        st.session_state.setdefault(notes_key, "")
        extra_notes = st.text_area(
            "Дополнительные требования / примечания",
            key=notes_key,
            placeholder="(опционально)",
        )

        if st.button("Скопировать и сохранить"):
            img_path = saved_path or st.session_state.get("last_uploaded_path")
            if not img_path:
                st.error("Пожалуйста, загрузите файл перед сохранением.")
            else:
                try:
                    # 10 параметров + примечания = 11 полей для CSV
                    collected = [
                        st.session_state.get(f"field_{i+1}", "") for i in range(10)
                    ]
                    collected.append(extra_notes or "")
                    collected = [v if v is not None else "" for v in collected]
                    save_submission(img_path, collected)
                    st.success("Заявка скопирована и сохранена.")
                    for i in range(10):
                        st.session_state.pop(f"field_{i+1}", None)
                    st.session_state.pop(notes_key, None)
                except Exception as e:
                    st.error(f"Ошибка при сохранении заявки: {e}")

    # ------------- Вкладка: Дашборд -------------

    with tab_dashboard:
        st.header("Дашборд заявок")
        submissions = read_submissions()
        st.write(f"Всего заявок: {len(submissions)}")

        if submissions:
            df = pd.DataFrame(submissions)

            if "timestamp" in df.columns:
                df["_ts_parsed"] = pd.to_datetime(df["timestamp"], errors="coerce")
                valid_ts = df["_ts_parsed"].dropna()
                if not valid_ts.empty:
                    min_date = valid_ts.min().date()
                    max_date = valid_ts.max().date()
                else:
                    min_date = max_date = None
            else:
                df["_ts_parsed"] = pd.NaT
                min_date = max_date = None

            st.write("**Фильтр по дате (UTC):**")
            col_from, col_to = st.columns(2)
            with col_from:
                start_date = st.date_input(
                    "От", value=min_date or datetime.date.today(), key="filter_from"
                )
            with col_to:
                end_date = st.date_input(
                    "До", value=max_date or datetime.date.today(), key="filter_to"
                )

            try:
                if min_date is not None:
                    mask = (df["_ts_parsed"].dt.date >= start_date) & (
                        df["_ts_parsed"].dt.date <= end_date
                    )
                    filtered = df[mask].copy()
                else:
                    filtered = df.copy()
            except Exception:
                filtered = df.copy()

            to_show = filtered.drop(
                columns=[c for c in ["_ts_parsed"] if c in filtered.columns]
            )
            st.dataframe(to_show, use_container_width=True)

            try:
                towrite = io.BytesIO()
                to_show.to_excel(
                    towrite,
                    index=False,
                    engine="openpyxl",
                )
                towrite.seek(0)
                st.download_button(
                    "Скачать Excel (.xlsx)",
                    data=towrite,
                    file_name="submissions.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            except Exception as e:
                st.warning(f"Не удалось сформировать Excel: {e}")
        else:
            st.info("Заявок ещё нет. Отправьте первую через вкладку 'Форма загрузки'.")


if __name__ == "__main__":
    main()
