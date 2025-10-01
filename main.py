# main.py
# -----------------------------------------------------------------------------
# Конвертер 16:9 → 1:1 → 9:16 + видео-баннер (FFmpeg, Tkinter, без внешних pip пакетов)
# -----------------------------------------------------------------------------
# Что умеет:
#  • Одиночный экспорт 60 сек ИЛИ пакет: весь ролик порезать по 60 сек
#  • Кадрирование 16:9 → 1:1 (центр / смещение / панорама L→R, R→L)
#  • Текстовый водяной знак (опц.)
#  • Видео-баннер: отдельный видеофайл, масштаб в %, позиция (углы/центр), отступ,
#    + опция «зациклить» (баннер крутится до конца клипа)
#  • Финальный вывод в 9:16 (1080x1920): квадрат по центру + чёрные поля
#  • Логи FFmpeg в консоль + <имя>.ffmpeg.log
# -----------------------------------------------------------------------------
# Требуется ffmpeg/ffprobe в PATH (macOS: brew install ffmpeg)
# Запуск: просто Run ▶ main.py в IDE
# -----------------------------------------------------------------------------

import os
import json
import shlex
import shutil
import math
import threading
import subprocess
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

APP_TITLE = "Конвертер 16:9 → 1:1 → 9:16 + видео-баннер (FFmpeg)"
OUTPUT_SQUARE = 1080          # 1:1 размер внутреннего квадрата
OUTPUT_W, OUTPUT_H = 1080, 1920  # финальный портрет 9:16
DEFAULT_FPS = 30
DEFAULT_CRF = 22
DEFAULT_VBITRATE = "6M"
DEFAULT_ABITRATE = "192k"
MARGIN_DEFAULT = 24

# -------------------------
# FFmpeg helpers
# -------------------------

def which_ffmpeg() -> str:
    p = shutil.which("ffmpeg")
    if not p:
        raise RuntimeError("Не найден ffmpeg в PATH. Установите его (macOS: brew install ffmpeg)")
    return p

def which_ffprobe() -> str:
    p = shutil.which("ffprobe")
    if not p:
        raise RuntimeError("Не найден ffprobe в PATH. Установите ffmpeg (brew install ffmpeg)")
    return p

def probe_video(path: str) -> dict:
    cmd = [
        which_ffprobe(),
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height:format=duration",
        "-of", "json",
        path,
    ]
    out = subprocess.check_output(cmd)
    data = json.loads(out.decode("utf-8"))
    w = data.get("streams", [{}])[0].get("width")
    h = data.get("streams", [{}])[0].get("height")
    dur = float(data.get("format", {}).get("duration", 0.0))
    return {"width": w, "height": h, "duration": dur}

# -------------------------
# Шрифт и экранирование для drawtext
# -------------------------

def detect_fontfile() -> str:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return ""

FONTFILE = detect_fontfile()

def fontfile_opt() -> str:
    return f":fontfile='{FONTFILE}'" if FONTFILE else ""

def esc_text_for_drawtext(txt: str) -> str:
    # https://ffmpeg.org/ffmpeg-filters.html#Notes-on-filtergraph-escaping
    txt = txt.replace("\\", "\\\\")
    txt = txt.replace(":", "\\:")
    txt = txt.replace("'", "\\'")
    txt = txt.replace("\n", "\\n")
    return txt

# -------------------------
# Построение filter_complex
# -------------------------

def build_filter_complex(
    mode: str, offset: float, dur: float,
    wm_text: str, wm_pos: str, wm_opacity: float,
    overlay_enabled: bool, overlay_scale_pct: int, overlay_pos: str, overlay_margin: int
) -> str:
    """
    Формируем единый filter_complex, который:
      [0:v] -> crop/scale до 1080x1080 -> (опц.) drawtext -> pad до 1080x1920 -> [base]
      (опц.) [1:v] -> scale (по % от ширины 1080) -> overlay на [base] в заданной позиции -> [outv]
    """
    ff = fontfile_opt()

    # 1) Кадрирование 16:9 → квадрат 1080x1080
    if mode == "center":
        crop = "crop=ih:ih:x=(iw-ih)/2:y=(ih-ih)/2"
    elif mode == "offset":
        crop = f"crop=ih:ih:x=(iw-ih)*{max(0.0, min(1.0, offset))}:y=(ih-ih)/2"
    elif mode == "pan_lr":
        crop = f"crop=ih:ih:x=(iw-ih)*(t/{max(dur, 0.001):.6f}):y=(ih-ih)/2"
    else:
        crop = f"crop=ih:ih:x=(iw-ih)*(1-(t/{max(dur, 0.001):.6f})):y=(ih-ih)/2"

    chain0 = f"[0:v]{crop},scale={OUTPUT_SQUARE}:{OUTPUT_SQUARE}"

    # 2) Текстовый водяной знак (опционально)
    if wm_text.strip():
        ttxt = esc_text_for_drawtext(wm_text.strip())
        if wm_pos == "right-bottom":
            x, y = f"w-text_w-{MARGIN_DEFAULT}", f"h-text_h-{MARGIN_DEFAULT}"
        elif wm_pos == "right-top":
            x, y = f"w-text_w-{MARGIN_DEFAULT}", f"{MARGIN_DEFAULT}"
        elif wm_pos == "left-bottom":
            x, y = f"{MARGIN_DEFAULT}", f"h-text_h-{MARGIN_DEFAULT}"
        elif wm_pos == "left-top":
            x, y = f"{MARGIN_DEFAULT}", f"{MARGIN_DEFAULT}"
        else:
            x, y = "(w-text_w)/2", "(h-text_h)/2"
        chain0 += (
            ",drawtext="
            f"text='{ttxt}'{ff}:fontcolor=white@{max(0.0, min(1.0, wm_opacity))}:fontsize=46:"
            f"box=1:boxcolor=black@0.35:boxborderw=10:x={x}:y={y}"
        )

    # 3) Паддинг квадрат → 1080x1920 (чёрные поля сверху/снизу)
    pad_x = (OUTPUT_W - OUTPUT_SQUARE) // 2  # 0
    pad_y = (OUTPUT_H - OUTPUT_SQUARE) // 2  # 420
    chain0 += f",pad={OUTPUT_W}:{OUTPUT_H}:{pad_x}:{pad_y}:color=black[base]"

    if not overlay_enabled:
        return f"{chain0};[base]copy[outv]"

    # 4) Видео-баннер
    ow = max(1, min(100, overlay_scale_pct)) * OUTPUT_W // 100
    chain1 = f"[1:v]setpts=PTS-STARTPTS,scale={ow}:-2[ov]"

    m = max(0, overlay_margin)
    if overlay_pos == "top-left":
        ox, oy = f"{m}", f"{m}"
    elif overlay_pos == "top-right":
        ox, oy = f"main_w-w-{m}", f"{m}"
    elif overlay_pos == "bottom-left":
        ox, oy = f"{m}", f"main_h-h-{m}"
    elif overlay_pos == "bottom-right":
        ox, oy = f"main_w-w-{m}", f"main_h-h-{m}"
    else:
        ox, oy = "(main_w-w)/2", "(main_h-h)/2"

    overlay = f"[base][ov]overlay=x={ox}:y={oy}:shortest=1[outv]"

    return f"{chain0};{chain1};{overlay}"

# -------------------------
# Команда FFmpeg
# -------------------------

def build_ffmpeg_cmd(
    in_path: str, out_path: str, start: float, duration: float,
    fps: int, crf: int,
    mode: str, offset: float,
    wm_text: str, wm_pos: str, wm_opacity: float,
    overlay_path: str, overlay_scale_pct: int, overlay_pos: str, overlay_margin: int,
    overlay_loop: bool
) -> list:
    ffmpeg = which_ffmpeg()

    fc = build_filter_complex(
        mode=mode, offset=offset, dur=duration,
        wm_text=wm_text, wm_pos=wm_pos, wm_opacity=wm_opacity,
        overlay_enabled=bool(overlay_path), overlay_scale_pct=overlay_scale_pct,
        overlay_pos=overlay_pos, overlay_margin=overlay_margin
    )

    cmd = [ffmpeg, "-y"]
    cmd += ["-ss", f"{max(0.0, start):.3f}", "-t", f"{max(0.0, duration):.3f}"]
    cmd += ["-i", in_path]

    if overlay_path:
        if overlay_loop:
            cmd += ["-stream_loop", "-1"]  # бесконечный цикл баннера
        cmd += ["-i", overlay_path]

    cmd += ["-filter_complex", fc, "-map", "[outv]"]
    cmd += ["-map", "0:a?"]  # аудио только из основного

    cmd += [
        "-r", str(fps),
        "-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-b:v", DEFAULT_VBITRATE,
        "-c:a", "aac", "-b:a", DEFAULT_ABITRATE,
        out_path
    ]
    return cmd

# -------------------------
# GUI
# -------------------------

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("900x840")

        self.in_path = None
        self.meta = {"width": 0, "height": 0, "duration": 0.0}

        # Режим
        self.export_mode = tk.StringVar(value="single")  # single|batch
        self.start_var = tk.DoubleVar(value=0.0)
        self.fps_var = tk.IntVar(value=DEFAULT_FPS)
        self.crf_var = tk.IntVar(value=DEFAULT_CRF)

        # Кадрирование
        self.mode_var = tk.StringVar(value="center")     # center|offset|pan_lr|pan_rl
        self.offset_var = tk.DoubleVar(value=0.5)

        # Водяной знак (текст)
        self.wm_text = tk.StringVar(value="@yourbrand")
        self.wm_pos = tk.StringVar(value="right-bottom")
        self.wm_opacity = tk.DoubleVar(value=0.85)

        # Видео-баннер (оверлей)
        self.overlay_path = tk.StringVar(value="")
        self.overlay_scale = tk.IntVar(value=30)
        self.overlay_pos = tk.StringVar(value="top-right")
        self.overlay_margin = tk.IntVar(value=MARGIN_DEFAULT)
        self.overlay_loop = tk.BooleanVar(value=True)  # зациклить баннер-видео

        self.status_var = tk.StringVar(value="Готов к работе")
        self._build_ui()

    def _build_ui(self):
        pad = dict(padx=8, pady=6)

        # Источник
        lf_src = ttk.LabelFrame(self, text="Источник видео")
        lf_src.pack(fill="x", **pad)
        ttk.Button(lf_src, text="Выбрать файл…", command=self._choose_file).pack(side="left", padx=6, pady=6)
        self.src_label = ttk.Label(lf_src, text="Файл не выбран")
        self.src_label.pack(side="left", padx=6)

        # Режим экспорта
        lf_mode = ttk.LabelFrame(self, text="Режим экспорта")
        lf_mode.pack(fill="x", **pad)
        ttk.Radiobutton(lf_mode, text="Один клип (60 сек)", value="single", variable=self.export_mode).pack(anchor="w")
        ttk.Radiobutton(lf_mode, text="Разбить весь ролик по 60 сек", value="batch", variable=self.export_mode).pack(anchor="w")

        # Фрагмент (для одиночного режима)
        lf_cut = ttk.LabelFrame(self, text="Фрагмент (только для одиночного режима)")
        lf_cut.pack(fill="x", **pad)
        ttk.Label(lf_cut, text="Старт (сек):").pack(side="left")
        self.sld_start = ttk.Scale(lf_cut, from_=0.0, to=0.0, orient="horizontal", variable=self.start_var)
        self.sld_start.pack(fill="x", expand=True, padx=8)
        self.lbl_dur = ttk.Label(lf_cut, text="Длительность: ?")
        self.lbl_dur.pack(side="right")

        # Кадрирование 1:1
        lf_crop = ttk.LabelFrame(self, text="Кадрирование 1:1")
        lf_crop.pack(fill="x", **pad)
        for text, val in [("Центр", "center"), ("Смещение", "offset"), ("Панорама L→R", "pan_lr"), ("Панорама R→L", "pan_rl")]:
            ttk.Radiobutton(lf_crop, text=text, value=val, variable=self.mode_var).pack(anchor="w")
        fr_off = ttk.Frame(lf_crop); fr_off.pack(fill="x")
        ttk.Label(fr_off, text="Положение (0…1)").pack(side="left")
        ttk.Scale(fr_off, from_=0.0, to=1.0, orient="horizontal", variable=self.offset_var).pack(fill="x", expand=True, padx=8)

        # Водяной знак (текст)
        lf_wm = ttk.LabelFrame(self, text="Текстовый водяной знак (опц.)")
        lf_wm.pack(fill="x", **pad)
        ttk.Entry(lf_wm, textvariable=self.wm_text, width=24).pack(side="left", padx=6)
        ttk.Label(lf_wm, text="Позиция").pack(side="left", padx=(8, 0))
        cbp = ttk.Combobox(lf_wm, state="readonly",
                           values=["right-bottom", "right-top", "left-bottom", "left-top", "center"], width=14)
        cbp.pack(side="left", padx=6)
        cbp.set(self.wm_pos.get())
        cbp.bind("<<ComboboxSelected>>", lambda e: self.wm_pos.set(cbp.get()))
        ttk.Label(lf_wm, text="Прозрачность").pack(side="left", padx=(8, 0))
        ttk.Scale(lf_wm, from_=0.1, to=1.0, orient="horizontal", variable=self.wm_opacity).pack(side="left", fill="x", expand=True, padx=8)

        # Видео-баннер (оверлей)
        lf_ov = ttk.LabelFrame(self, text="Видео-баннер (оверлей)")
        lf_ov.pack(fill="x", **pad)
        ttk.Button(lf_ov, text="Выбрать видео баннера…", command=self._choose_overlay).pack(side="left", padx=6, pady=6)
        self.overlay_label = ttk.Label(lf_ov, text="Не выбрано")
        self.overlay_label.pack(side="left", padx=6)
        fr_ov2 = ttk.Frame(lf_ov); fr_ov2.pack(fill="x", pady=(6,0))
        ttk.Label(fr_ov2, text="Размер баннера, % от ширины кадра (1080):").pack(side="left")
        ttk.Spinbox(fr_ov2, from_=10, to=100, textvariable=self.overlay_scale, width=5).pack(side="left", padx=6)
        ttk.Label(fr_ov2, text="Позиция").pack(side="left", padx=(12, 4))
        cbpos = ttk.Combobox(fr_ov2, state="readonly",
                             values=["top-left","top-right","bottom-left","bottom-right","center"], width=14)
        cbpos.pack(side="left")
        cbpos.set(self.overlay_pos.get())
        cbpos.bind("<<ComboboxSelected>>", lambda e: self.overlay_pos.set(cbpos.get()))
        ttk.Label(fr_ov2, text="Отступ (px)").pack(side="left", padx=(12, 4))
        ttk.Spinbox(fr_ov2, from_=0, to=200, textvariable=self.overlay_margin, width=6).pack(side="left")
        ttk.Checkbutton(lf_ov, text="Зациклить баннер до конца клипа", variable=self.overlay_loop).pack(anchor="w", padx=6, pady=(6,0))

        # Экспорт
        lf_exp = ttk.LabelFrame(self, text="Экспорт")
        lf_exp.pack(fill="x", **pad)
        ttk.Label(lf_exp, text="FPS").pack(side="left")
        ttk.Spinbox(lf_exp, from_=24, to=60, textvariable=self.fps_var, width=5).pack(side="left", padx=6)
        ttk.Label(lf_exp, text="CRF").pack(side="left")
        ttk.Spinbox(lf_exp, from_=18, to=30, textvariable=self.crf_var, width=5).pack(side="left", padx=6)
        ttk.Button(lf_exp, text="Запустить экспорт", command=self.convert_async).pack(side="right", padx=6)

        ttk.Label(self, textvariable=self.status_var, anchor="w").pack(fill="x", padx=8, pady=8)

    # --- Actions

    def _choose_file(self):
        p = filedialog.askopenfilename(title="Выберите видео", filetypes=[
            ("Видео", "*.mp4 *.mov *.mkv *.avi *.m4v"), ("Все файлы", "*.*")
        ])
        if not p:
            return
        self.in_path = p
        self.src_label.config(text=os.path.basename(p))
        self.status_var.set("Чтение метаданных…")
        self.update_idletasks()
        try:
            self.meta = probe_video(self.in_path)
            total = float(self.meta.get("duration", 0.0))
            self.sld_start.configure(to=max(0.0, total - 60.0))
            self.start_var.set(0.0)
            self.lbl_dur.config(text=f"Длительность исходника: {seconds_to_hms(total)}")
            self.status_var.set("Файл загружен. Готово к экспорту.")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось прочитать метаданные: {e}")
            self.status_var.set("Ошибка при чтении файла")

    def _choose_overlay(self):
        p = filedialog.askopenfilename(title="Выберите видео-баннер", filetypes=[
            ("Видео", "*.mp4 *.mov *.mkv *.avi *.m4v"), ("Все файлы", "*.*")
        ])
        if p:
            self.overlay_path.set(p)
            self.overlay_label.config(text=os.path.basename(p))

    def convert_async(self):
        if not self.in_path:
            messagebox.showwarning("Внимание", "Сначала выберите видеофайл.")
            return
        t = threading.Thread(target=self._convert_safe)
        t.daemon = True
        t.start()

    def _convert_safe(self):
        try:
            if self.export_mode.get() == "batch":
                self._convert_batch()
            else:
                self._convert_single()
        except Exception as e:
            print("[ERROR]", e)
            messagebox.showerror("Ошибка", "Произошла ошибка. Полный текст в консоли/логе .ffmpeg.log")
            self.status_var.set("Ошибка при конвертации")

    def _convert_single(self):
        out_path = filedialog.asksaveasfilename(
            title="Сохранить как", defaultextension=".mp4",
            filetypes=[("MP4", "*.mp4")], initialfile="output_9x16.mp4"
        )
        if not out_path:
            return

        start = float(self.start_var.get())
        total = float(self.meta.get("duration", 0.0))
        dur = 60.0 if total >= 60.0 else max(0.1, total - start)

        cmd = build_ffmpeg_cmd(
            in_path=self.in_path,
            out_path=out_path,
            start=start,
            duration=dur,
            fps=int(self.fps_var.get()),
            crf=int(self.crf_var.get()),
            mode=self.mode_var.get(),
            offset=float(self.offset_var.get()),
            wm_text=self.wm_text.get(),
            wm_pos=self.wm_pos.get(),
            wm_opacity=float(self.wm_opacity.get()),
            overlay_path=self.overlay_path.get(),
            overlay_scale_pct=int(self.overlay_scale.get()),
            overlay_pos=self.overlay_pos.get(),
            overlay_margin=int(self.overlay_margin.get()),
            overlay_loop=bool(self.overlay_loop.get()),
        )
        self._run_ffmpeg(cmd, out_path)

    def _convert_batch(self):
        out_dir = filedialog.askdirectory(title="Выберите папку для клипов")
        if not out_dir:
            return
        base = os.path.splitext(os.path.basename(self.in_path))[0]

        total = float(self.meta.get("duration", 0.0))
        n_clips = int(math.ceil(total / 60.0)) if total > 0 else 0
        if n_clips == 0:
            messagebox.showwarning("Внимание", "Не удалось определить длительность исходника.")
            return

        self.status_var.set(f"Пакетный экспорт: {n_clips} клипов по 60 сек…")
        self.update_idletasks()

        for i in range(n_clips):
            start = i * 60.0
            remain = max(0.0, total - start)
            dur = 60.0 if remain >= 60.0 else remain
            dur = max(0.1, dur)

            out_name = f"{base}_part_{i+1:02d}_9x16.mp4"
            out_path = os.path.join(out_dir, out_name)

            cmd = build_ffmpeg_cmd(
                in_path=self.in_path,
                out_path=out_path,
                start=start,
                duration=dur,
                fps=int(self.fps_var.get()),
                crf=int(self.crf_var.get()),
                mode=self.mode_var.get(),
                offset=float(self.offset_var.get()),
                wm_text=self.wm_text.get(),
                wm_pos=self.wm_pos.get(),
                wm_opacity=float(self.wm_opacity.get()),
                overlay_path=self.overlay_path.get(),
                overlay_scale_pct=int(self.overlay_scale.get()),
                overlay_pos=self.overlay_pos.get(),
                overlay_margin=int(self.overlay_margin.get()),
                overlay_loop=bool(self.overlay_loop.get()),
            )

            self.status_var.set(f"Экспорт {i+1}/{n_clips}: {out_name}")
            self.update_idletasks()
            self._run_ffmpeg(cmd, out_path, silent=True)

        self.status_var.set(f"Готово: создано {n_clips} клипов в {out_dir}")
        messagebox.showinfo("Готово", f"Создано {n_clips} файлов по 60 сек.\nПапка:\n{out_dir}")

    def _run_ffmpeg(self, cmd, out_path, silent=False):
        printable = " ".join(shlex.quote(x) for x in cmd)
        log_path = os.path.splitext(out_path)[0] + ".ffmpeg.log"
        print("\n[FFMPEG CMD]", printable)
        try:
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  text=True, encoding="utf-8", errors="replace")
            with open(log_path, "w", encoding="utf-8") as f:
                f.write("[FFMPEG CMD] " + printable + "\n\n")
                f.write(proc.stderr or "")
            if proc.returncode != 0:
                print("[FFMPEG EXIT]", proc.returncode)
                print("[FFMPEG STDERR]\n" + (proc.stderr or ""))
                raise RuntimeError(f"FFmpeg завершился с ошибкой. Лог: {log_path}")
            if not silent:
                self.status_var.set(f"Готово: {os.path.basename(out_path)}")
                print("[OK] Saved:", out_path)
                messagebox.showinfo("Готово", f"Файл сохранён:\n{out_path}\nЛог: {log_path}")
        except FileNotFoundError:
            print("[ERROR] ffmpeg not found")
            raise RuntimeError("FFmpeg не найден. Установите его и убедитесь, что он в PATH.")

# -------------------------
# Вспомогательное
# -------------------------

def seconds_to_hms(t: float) -> str:
    t = max(0, int(t))
    h = t // 3600
    m = (t % 3600) // 60
    s = t % 60
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

if __name__ == "__main__":
    try:
        which_ffmpeg(); which_ffprobe()
    except Exception as e:
        messagebox.showerror(APP_TITLE, str(e))
        raise
    app = App()
    app.mainloop()
