import os
import sqlite3
import tkinter as tk
from dataclasses import dataclass
from datetime import datetime
from tkinter import filedialog, messagebox, ttk

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

try:
    import sv_ttk  # type: ignore
except (ModuleNotFoundError, ImportError):
    sv_ttk = None

DB_PATH = "gestao_pecuaria_v3.db"
DATE_FMT = "%d/%m/%Y"
DATE_TIME_FMT = "%d/%m/%Y %H:%M"


@dataclass(frozen=True)
class AnimalFormData:
    n_animal: str
    raca: str
    cor: str
    dt_com: str
    val_com: float
    peso_ini: float
    tipo_confi: str
    val_arr_venda: float
    dt_vend: str
    desc_carac: str


class Database:
    def __init__(self, db_path: str = DB_PATH) -> None:
        self.db_path = db_path

    def get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    def init_db(self) -> None:
        with self.get_conn() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS animais (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    n_animal TEXT NOT NULL,
                    raca TEXT NOT NULL,
                    cor TEXT,
                    dt_com TEXT NOT NULL,
                    val_com REAL NOT NULL DEFAULT 0,
                    peso_ini REAL NOT NULL DEFAULT 0,
                    tipo_confi TEXT NOT NULL,
                    val_arr_venda REAL NOT NULL DEFAULT 0,
                    dt_vend TEXT,
                    desc_carac TEXT,
                    foto_path TEXT
                );

                CREATE TABLE IF NOT EXISTS vacinas (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    id_animal INTEGER NOT NULL,
                    nome_vacina TEXT NOT NULL,
                    dt_vencimento TEXT NOT NULL,
                    FOREIGN KEY (id_animal) REFERENCES animais(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS historico (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    id_animal INTEGER NOT NULL,
                    tipo_evento TEXT NOT NULL,
                    data TEXT NOT NULL,
                    descricao TEXT NOT NULL,
                    FOREIGN KEY (id_animal) REFERENCES animais(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_animais_n_animal ON animais(n_animal);
                CREATE INDEX IF NOT EXISTS idx_historico_id_animal ON historico(id_animal);
                CREATE INDEX IF NOT EXISTS idx_vacinas_id_animal ON vacinas(id_animal);
                """
            )


class AppPecuariaCRUD:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Sistema Pecuária Pro - CRUD Completo")
        self.root.geometry("1300x850")
        self.root.minsize(1100, 700)

        self.db = Database(DB_PATH)
        self.db.init_db()

        self.taxas_gmd = {"Pasto": 0.45, "Semi-pasto": 0.85, "Confinado": 1.50}
        self.selected_id: int | None = None
        self.foto_path_atual: str | None = None

        self.setup_ui()
        self.atualizar_tabela()

    @staticmethod
    def parse_money(value: str, field_name: str) -> float:
        value = value.strip().replace(".", "").replace(",", ".")
        if not value:
            return 0.0
        try:
            return float(value)
        except ValueError as exc:
            raise ValueError(f"Campo '{field_name}' inválido.") from exc

    @staticmethod
    def validate_date(value: str, field_name: str, optional: bool = False) -> str:
        value = value.strip()
        if optional and not value:
            return ""
        try:
            datetime.strptime(value, DATE_FMT)
            return value
        except ValueError as exc:
            raise ValueError(f"Campo '{field_name}' deve estar no formato DD/MM/AAAA.") from exc

    def registrar_evento(self, conn: sqlite3.Connection, id_animal: int, tipo: str, descricao: str) -> None:
        conn.execute(
            "INSERT INTO historico (id_animal, tipo_evento, data, descricao) VALUES (?,?,?,?)",
            (id_animal, tipo, datetime.now().strftime(DATE_TIME_FMT), descricao),
        )

    def setup_ui(self) -> None:
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True)

        self.aba1 = ttk.Frame(self.notebook)
        self.aba2 = ttk.Frame(self.notebook)
        self.aba3 = ttk.Frame(self.notebook)
        self.aba4 = ttk.Frame(self.notebook)

        self.notebook.add(self.aba1, text="Gestão de Animais")
        self.notebook.add(self.aba2, text="Vacinas")
        self.notebook.add(self.aba3, text="Dashboard")
        self.notebook.add(self.aba4, text="Histórico")

        self.setup_aba_gestao()
        self.setup_aba_vacinas()
        self.setup_aba_dashboard()
        self.setup_aba_historico()

    def setup_aba_gestao(self) -> None:
        frame = ttk.LabelFrame(self.aba1, text="Ficha do Animal")
        frame.pack(fill="x", padx=10, pady=5)
        frame.columnconfigure((0, 1, 2), weight=1)

        campos = [
            ("N° Animal", "n_animal"),
            ("Raça", "raca"),
            ("Cor", "cor"),
            ("Data Compra", "dt_com"),
            ("Valor Compra", "val_com"),
            ("Peso Inicial", "peso_ini"),
            ("Tipo Manejo", "tipo_confi"),
            ("Preço Arroba", "val_arr_venda"),
            ("Data Venda", "dt_vend"),
            ("Descrição", "desc_carac"),
        ]

        self.inputs: dict[str, ttk.Entry | ttk.Combobox] = {}

        for i, (label, key) in enumerate(campos):
            row, col = divmod(i, 3)
            ttk.Label(frame, text=label).grid(row=row * 2, column=col, sticky="w")
            if key == "tipo_confi":
                ent = ttk.Combobox(frame, values=list(self.taxas_gmd.keys()), state="readonly")
                ent.set("Pasto")
            else:
                ent = ttk.Entry(frame)
            ent.grid(row=row * 2 + 1, column=col, sticky="ew", padx=5, pady=3)
            self.inputs[key] = ent

        btns = ttk.Frame(self.aba1)
        btns.pack(pady=10)

        ttk.Button(btns, text="💾 Salvar", command=self.db_salvar_animal).pack(side="left", padx=5)
        ttk.Button(btns, text="✨ Novo", command=self.limpar_campos).pack(side="left", padx=5)
        ttk.Button(btns, text="🗑️ Excluir", command=self.db_excluir_animal).pack(side="left", padx=5)
        ttk.Button(btns, text="📸 Foto", command=self.selecionar_foto).pack(side="left", padx=5)

        self.lbl_foto = ttk.Label(self.aba1, text="Nenhuma foto selecionada")
        self.lbl_foto.pack()

        self.tree = ttk.Treeview(
            self.aba1,
            columns=("ID", "N", "Raça", "Manejo", "Peso", "@", "Lucro"),
            show="headings",
        )
        for col in self.tree["columns"]:
            self.tree.heading(col, text=col)
        self.tree.pack(fill="both", expand=True, padx=10, pady=10)

        self.tree.bind("<<TreeviewSelect>>", self.carregar_campos_selecionados)

    def selecionar_foto(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Imagens", "*.png;*.jpg;*.jpeg")])
        if path:
            self.foto_path_atual = path
            self.lbl_foto.config(text=os.path.basename(path))

    def setup_aba_vacinas(self) -> None:
        frame = ttk.LabelFrame(self.aba2, text="Registrar Vacina")
        frame.pack(fill="x", padx=10, pady=5)

        ttk.Label(frame, text="ID Animal").grid(row=0, column=0)
        ttk.Label(frame, text="Vacina").grid(row=0, column=1)
        ttk.Label(frame, text="Vencimento").grid(row=0, column=2)

        self.vac_animal = ttk.Entry(frame)
        self.vac_nome = ttk.Entry(frame)
        self.vac_venc = ttk.Entry(frame)

        self.vac_animal.grid(row=1, column=0, padx=5)
        self.vac_nome.grid(row=1, column=1, padx=5)
        self.vac_venc.grid(row=1, column=2, padx=5)

        ttk.Button(frame, text="💉 Registrar", command=self.db_salvar_vacina).grid(row=1, column=3, padx=10)

        self.tree_vac = ttk.Treeview(
            self.aba2,
            columns=("ID", "Animal", "Vacina", "Vencimento", "Status"),
            show="headings",
        )
        for col in self.tree_vac["columns"]:
            self.tree_vac.heading(col, text=col)
        self.tree_vac.pack(fill="both", expand=True, padx=10, pady=10)

        ttk.Button(self.aba2, text="🗑️ Remover", command=self.db_excluir_vacina).pack()

    def setup_aba_historico(self) -> None:
        top = ttk.Frame(self.aba4)
        top.pack(fill="x", padx=10, pady=5)

        ttk.Label(top, text="ID ou Número do Animal:").pack(side="left")
        self.hist_busca = ttk.Entry(top, width=15)
        self.hist_busca.pack(side="left", padx=5)
        ttk.Button(top, text="Buscar", command=self.carregar_historico).pack(side="left")

        self.tree_hist = ttk.Treeview(
            self.aba4,
            columns=("ID", "Evento", "Data", "Descrição"),
            show="headings",
        )
        for col in self.tree_hist["columns"]:
            self.tree_hist.heading(col, text=col)
        self.tree_hist.pack(fill="both", expand=True, padx=10, pady=10)

    def carregar_historico(self) -> None:
        for i in self.tree_hist.get_children():
            self.tree_hist.delete(i)

        chave = self.hist_busca.get().strip()
        if not chave:
            return

        with self.db.get_conn() as conn:
            rows = conn.execute(
                """
                SELECT h.id, h.tipo_evento, h.data, h.descricao
                FROM historico h
                JOIN animais a ON a.id = h.id_animal
                WHERE CAST(a.id AS TEXT) = ? OR a.n_animal = ?
                ORDER BY h.id DESC
                """,
                (chave, chave),
            ).fetchall()

        for row in rows:
            self.tree_hist.insert("", "end", values=row)

    def limpar_campos(self) -> None:
        self.selected_id = None
        self.foto_path_atual = None
        for key, ent in self.inputs.items():
            ent.delete(0, tk.END)
            if key == "tipo_confi":
                ent.insert(0, "Pasto")
        self.lbl_foto.config(text="Nenhuma foto selecionada")

    def carregar_campos_selecionados(self, _event: tk.Event) -> None:
        sel = self.tree.selection()
        if not sel:
            return

        item = self.tree.item(sel[0])
        if not item.get("values"):
            return

        id_animal = int(item["values"][0])
        self.selected_id = id_animal

        with self.db.get_conn() as conn:
            dados = conn.execute("SELECT * FROM animais WHERE id=?", (id_animal,)).fetchone()

        if not dados:
            return

        keys = [
            "id",
            "n_animal",
            "raca",
            "cor",
            "dt_com",
            "val_com",
            "peso_ini",
            "tipo_confi",
            "val_arr_venda",
            "dt_vend",
            "desc_carac",
            "foto_path",
        ]
        row = dict(zip(keys, dados))

        for key in self.inputs:
            self.inputs[key].delete(0, tk.END)
            if row.get(key) is not None:
                self.inputs[key].insert(0, str(row[key]))

        self.foto_path_atual = row.get("foto_path")
        self.lbl_foto.config(text=os.path.basename(self.foto_path_atual) if self.foto_path_atual else "Nenhuma foto")

    def _collect_animal_data(self) -> AnimalFormData:
        raw = {k: v.get().strip() for k, v in self.inputs.items()}

        if not raw["n_animal"]:
            raise ValueError("Informe o número do animal.")
        if not raw["raca"]:
            raise ValueError("Informe a raça.")
        if not raw["tipo_confi"]:
            raise ValueError("Selecione o tipo de manejo.")

        return AnimalFormData(
            n_animal=raw["n_animal"],
            raca=raw["raca"],
            cor=raw["cor"],
            dt_com=self.validate_date(raw["dt_com"], "Data Compra"),
            val_com=self.parse_money(raw["val_com"], "Valor Compra"),
            peso_ini=self.parse_money(raw["peso_ini"], "Peso Inicial"),
            tipo_confi=raw["tipo_confi"],
            val_arr_venda=self.parse_money(raw["val_arr_venda"], "Preço Arroba"),
            dt_vend=self.validate_date(raw["dt_vend"], "Data Venda", optional=True),
            desc_carac=raw["desc_carac"],
        )

    def db_salvar_animal(self) -> None:
        try:
            data = self._collect_animal_data()
        except ValueError as exc:
            messagebox.showwarning("Validação", str(exc))
            return

        try:
            with self.db.get_conn() as conn:
                if self.selected_id is None:
                    cursor = conn.execute(
                        """
                        INSERT INTO animais
                        (n_animal, raca, cor, dt_com, val_com, peso_ini, tipo_confi, val_arr_venda, dt_vend, desc_carac, foto_path)
                        VALUES (?,?,?,?,?,?,?,?,?,?,?)
                        """,
                        (
                            data.n_animal,
                            data.raca,
                            data.cor,
                            data.dt_com,
                            data.val_com,
                            data.peso_ini,
                            data.tipo_confi,
                            data.val_arr_venda,
                            data.dt_vend,
                            data.desc_carac,
                            self.foto_path_atual,
                        ),
                    )
                    novo_id = int(cursor.lastrowid)
                    self.registrar_evento(conn, novo_id, "Cadastro", "Animal cadastrado.")
                    messagebox.showinfo("OK", "Animal cadastrado com sucesso!")
                else:
                    conn.execute(
                        """
                        UPDATE animais
                        SET n_animal=?, raca=?, cor=?, dt_com=?, val_com=?, peso_ini=?,
                            tipo_confi=?, val_arr_venda=?, dt_vend=?, desc_carac=?, foto_path=?
                        WHERE id=?
                        """,
                        (
                            data.n_animal,
                            data.raca,
                            data.cor,
                            data.dt_com,
                            data.val_com,
                            data.peso_ini,
                            data.tipo_confi,
                            data.val_arr_venda,
                            data.dt_vend,
                            data.desc_carac,
                            self.foto_path_atual,
                            self.selected_id,
                        ),
                    )
                    self.registrar_evento(conn, int(self.selected_id), "Atualização", "Dados atualizados.")
                    messagebox.showinfo("OK", "Animal atualizado com sucesso!")

            self.atualizar_tabela()
            self.limpar_campos()

        except sqlite3.DatabaseError as exc:
            messagebox.showerror("Erro no banco", str(exc))

    def db_excluir_animal(self) -> None:
        if self.selected_id is None:
            messagebox.showwarning("Aviso", "Selecione um animal.")
            return

        if not messagebox.askyesno("Confirmar", "Excluir animal selecionado?"):
            return

        try:
            with self.db.get_conn() as conn:
                self.registrar_evento(conn, int(self.selected_id), "Exclusão", "Animal removido.")
                conn.execute("DELETE FROM animais WHERE id=?", (self.selected_id,))

            self.atualizar_tabela()
            self.limpar_campos()
        except sqlite3.DatabaseError as exc:
            messagebox.showerror("Erro no banco", str(exc))

    def db_salvar_vacina(self) -> None:
        try:
            id_animal = int(self.vac_animal.get().strip())
            nome = self.vac_nome.get().strip()
            venc = self.validate_date(self.vac_venc.get(), "Vencimento")
            if not nome:
                raise ValueError("Informe o nome da vacina.")
        except ValueError as exc:
            messagebox.showwarning("Validação", str(exc))
            return

        try:
            with self.db.get_conn() as conn:
                exists = conn.execute("SELECT 1 FROM animais WHERE id=?", (id_animal,)).fetchone()
                if not exists:
                    messagebox.showwarning("Validação", "ID de animal não encontrado.")
                    return

                conn.execute(
                    "INSERT INTO vacinas (id_animal, nome_vacina, dt_vencimento) VALUES (?,?,?)",
                    (id_animal, nome, venc),
                )
                self.registrar_evento(conn, id_animal, "Vacina", f"Vacina '{nome}' registrada.")

            self.vac_nome.delete(0, tk.END)
            self.vac_venc.delete(0, tk.END)
            self.atualizar_tabela_vacinas()
        except sqlite3.DatabaseError as exc:
            messagebox.showerror("Erro no banco", str(exc))

    def db_excluir_vacina(self) -> None:
        sel = self.tree_vac.selection()
        if not sel:
            messagebox.showwarning("Aviso", "Selecione uma vacina para remover.")
            return

        values = self.tree_vac.item(sel[0]).get("values", [])
        if not values:
            return

        vac_id, id_animal = int(values[0]), int(values[1])
        try:
            with self.db.get_conn() as conn:
                conn.execute("DELETE FROM vacinas WHERE id=?", (vac_id,))
                self.registrar_evento(conn, id_animal, "Vacina", "Registro de vacina removido.")
            self.atualizar_tabela_vacinas()
        except sqlite3.DatabaseError as exc:
            messagebox.showerror("Erro no banco", str(exc))

    def atualizar_tabela(self) -> None:
        for i in self.tree.get_children():
            self.tree.delete(i)

        with self.db.get_conn() as conn:
            df = pd.read_sql_query("SELECT * FROM animais ORDER BY id DESC", conn)

        hoje = datetime.now()
        for _, row in df.iterrows():
            try:
                dias = (hoje - datetime.strptime(str(row["dt_com"]), DATE_FMT)).days
                gmd = self.taxas_gmd.get(str(row["tipo_confi"]), 0)
                peso = float(row["peso_ini"] or 0) + max(dias, 0) * gmd
                arroba = peso / 30
                preco = float(row["val_arr_venda"] or 0)
                custo = float(row["val_com"] or 0)
                lucro = arroba * preco - custo
            except (ValueError, TypeError):
                peso = arroba = lucro = 0

            self.tree.insert(
                "",
                "end",
                values=(
                    row["id"],
                    row["n_animal"],
                    row["raca"],
                    row["tipo_confi"],
                    f"{peso:.1f} kg",
                    f"{arroba:.2f} @",
                    f"R$ {lucro:.2f}",
                ),
            )

        self.atualizar_tabela_vacinas()

    def atualizar_tabela_vacinas(self) -> None:
        for i in self.tree_vac.get_children():
            self.tree_vac.delete(i)

        with self.db.get_conn() as conn:
            df = pd.read_sql_query("SELECT * FROM vacinas ORDER BY id DESC", conn)

        hoje = datetime.now()
        for _, row in df.iterrows():
            try:
                venc = datetime.strptime(str(row["dt_vencimento"]), DATE_FMT)
                status = "🟢 OK" if venc >= hoje else "🔴 VENCIDA"
            except ValueError:
                status = "❓ Data inválida"

            self.tree_vac.insert(
                "",
                "end",
                values=(row["id"], row["id_animal"], row["nome_vacina"], row["dt_vencimento"], status),
            )

    def setup_aba_dashboard(self) -> None:
        ttk.Button(self.aba3, text="Atualizar Dashboard", command=self.gerar_dashboard).pack(pady=10)
        self.dash_container = ttk.Frame(self.aba3)
        self.dash_container.pack(fill="both", expand=True)

    def gerar_dashboard(self) -> None:
        for widget in self.dash_container.winfo_children():
            widget.destroy()

        with self.db.get_conn() as conn:
            df = pd.read_sql_query("SELECT * FROM animais", conn)

        if df.empty:
            ttk.Label(self.dash_container, text="Sem dados para exibir").pack()
            return

        hoje = datetime.now()
        investido = float(df["val_com"].fillna(0).sum())
        valor_atual = 0.0

        for _, row in df.iterrows():
            try:
                dias = (hoje - datetime.strptime(str(row["dt_com"]), DATE_FMT)).days
                peso = float(row["peso_ini"] or 0) + max(dias, 0) * self.taxas_gmd.get(str(row["tipo_confi"]), 0)
                valor_atual += (peso / 30) * float(row["val_arr_venda"] or 0)
            except (ValueError, TypeError):
                continue

        kpi_frame = ttk.Frame(self.dash_container)
        kpi_frame.grid(row=0, column=0, sticky="ew", pady=5)
        kpi_frame.columnconfigure((0, 1, 2), weight=1)

        def kpi(col: int, label: str, value: str) -> None:
            box = ttk.Frame(kpi_frame, padding=10)
            box.grid(row=0, column=col, sticky="ew", padx=5)
            ttk.Label(box, text=label, font=("Segoe UI", 10, "bold")).pack(anchor="w")
            ttk.Label(box, text=value, font=("Segoe UI", 14)).pack(anchor="w")

        kpi(0, "Total de Animais", str(len(df)))
        kpi(1, "Investimento Total (R$)", f"{investido:,.2f}")
        kpi(2, "Valor Atual Estimado (R$)", f"{valor_atual:,.2f}")

        fig = plt.Figure(figsize=(11, 4), dpi=100)
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        ax1.bar(["Investimento", "Valor Atual"], [investido, valor_atual], color=["#3498db", "#2ecc71"])
        ax1.set_title("Saúde Financeira do Lote (R$)")
        ax1.grid(axis="y", linestyle="--", alpha=0.7)

        manejo_counts = df["tipo_confi"].value_counts()
        if not manejo_counts.empty:
            ax2.pie(
                manejo_counts,
                labels=manejo_counts.index,
                autopct="%1.1f%%",
                startangle=140,
                colors=["#f1c40f", "#e67e22", "#e74c3c"],
            )
            ax2.set_title("Distribuição por Tipo de Manejo")

        canvas = FigureCanvasTkAgg(fig, master=self.dash_container)
        canvas.draw()
        canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew", pady=5)


if __name__ == "__main__":
    root = tk.Tk()
    if sv_ttk:
        sv_ttk.set_theme("dark")
    app = AppPecuariaCRUD(root)
    root.mainloop()
