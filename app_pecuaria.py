import os
import sqlite3
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_pdf import PdfPages

try:
    from PIL import Image, ImageTk  # type: ignore
except (ModuleNotFoundError, ImportError):
    Image = None
    ImageTk = None

# Configuração do banco de dados
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "gestao_pecuaria_v3.db")

try:
    import sv_ttk
except (ModuleNotFoundError, ImportError):
    sv_ttk = None

DATE_FMT = "%d/%m/%Y"


class Database:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path

    def get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    def init_db(self):
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
                    status TEXT NOT NULL DEFAULT 'ATIVO',
                    desc_carac TEXT,
                    foto_path TEXT
                );

                CREATE TABLE IF NOT EXISTS vacinas (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    id_animal INTEGER NOT NULL,
                    nome_vacina TEXT NOT NULL,
                    dt_vencimento TEXT NOT NULL,
                    tipo_vacina TEXT DEFAULT 'Não Informado',
                    dt_aplicacao TEXT DEFAULT '',
                    FOREIGN KEY (id_animal) REFERENCES animais(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS historico (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    id_animal INTEGER,
                    tipo_evento TEXT NOT NULL,
                    data TEXT NOT NULL,
                    descricao TEXT NOT NULL,
                    origem TEXT DEFAULT 'geral',
                    FOREIGN KEY (id_animal) REFERENCES animais(id) ON DELETE CASCADE
                );
                """
            )

            # Migração defensiva de colunas antigas
            for ddl in (
                "ALTER TABLE animais ADD COLUMN foto_path TEXT",
                "ALTER TABLE vacinas ADD COLUMN tipo_vacina TEXT DEFAULT 'Não Informado'",
                "ALTER TABLE vacinas ADD COLUMN dt_aplicacao TEXT DEFAULT ''",
                "ALTER TABLE historico ADD COLUMN origem TEXT DEFAULT 'geral'",
                "ALTER TABLE animais ADD COLUMN status TEXT NOT NULL DEFAULT 'ATIVO'",
            ):
                try:
                    conn.execute(ddl)
                except sqlite3.OperationalError:
                    pass

            # Migração de compatibilidade: registros antigos com data de venda viram VENDIDO
            conn.execute(
                """
                UPDATE animais
                SET status = 'VENDIDO'
                WHERE (status IS NULL OR status = '' OR status = 'ATIVO')
                  AND dt_vend IS NOT NULL
                  AND TRIM(dt_vend) <> ''
                """
            )


class AppPecuariaCRUD:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Pecuária Pro - Gestão Precisa e Sanitária")
        self.root.geometry("1350x850")

        self.db = Database()
        self.db.init_db()

        self.taxas_gmd = {"Pasto": 0.45, "Semi-pasto": 0.85, "Confinado": 1.50}
        self.tipos_vacina = ["Aftosa", "Brucelose", "Raiva", "Clostridiose", "Vermífugo", "Carrapaticida", "Outros"]

        self.selected_id = None
        self.vac_selected_id = None
        self.foto_path_atual = None
        self._foto_preview_ref = None

        self.setup_ui()
        self.atualizar_tabela()
        self.atualizar_tabela_vacinas()
        self.gerar_dashboard()

    def registrar_evento(self, conn: sqlite3.Connection, id_animal: int, tipo: str, descricao: str, origem: str = "geral"):
        conn.execute(
            "INSERT INTO historico (id_animal, tipo_evento, data, descricao, origem) VALUES (?,?,?,?,?)",
            (id_animal, tipo, datetime.now().strftime("%d/%m/%Y %H:%M"), descricao, origem),
        )

    def parse_date(self, value: str, required: bool = True) -> bool:
        txt = (value or "").strip()
        if not txt:
            return not required
        try:
            datetime.strptime(txt, DATE_FMT)
            return True
        except ValueError:
            return False

    def parse_float(self, val):
        txt = str(val).replace("R$", "").strip()
        if not txt:
            return 0.0
        if "." in txt and "," in txt:
            txt = txt.replace(".", "").replace(",", ".")
        elif "," in txt:
            txt = txt.replace(",", ".")
        try:
            return float(txt)
        except ValueError:
            return 0.0

    def format_moeda(self, valor):
        return f"R$ {valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

    def setup_ui(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True)

        self.aba1 = ttk.Frame(self.notebook)
        self.aba2 = ttk.Frame(self.notebook)
        self.aba3 = ttk.Frame(self.notebook)
        self.aba4 = ttk.Frame(self.notebook)

        self.notebook.add(self.aba1, text="🐂 Gestão de Animais")
        self.notebook.add(self.aba2, text="💉 Controle de Vacinas")
        self.notebook.add(self.aba3, text="📊 Dashboard Geral")
        self.notebook.add(self.aba4, text="✅ Animais Vendidos")

        self.setup_aba_gestao()
        self.setup_aba_vacinas()
        self.setup_aba_dashboard()
        self.setup_aba_vendidos()

    def setup_aba_gestao(self):
        frame = ttk.LabelFrame(self.aba1, text="Ficha Técnica do Animal")
        frame.pack(fill="x", padx=10, pady=5)

        self.inputs = {}
        campos = [
            ("N° Animal", "n_animal"), ("Raça", "raca"), ("Peso Inicial (kg)", "peso_ini"),
            ("Data Compra", "dt_com"), ("Valor Compra", "val_com"), ("Manejo", "tipo_confi"),
            ("Valor @ Venda", "val_arr_venda"), ("Data Venda", "dt_vend"), ("Cor", "cor"),
            ("Descrição", "desc_carac")
        ]

        for i, (label, key) in enumerate(campos):
            r, c = divmod(i, 3)
            ttk.Label(frame, text=label).grid(row=r * 2, column=c, sticky="w", padx=5)
            if key == "tipo_confi":
                ent = ttk.Combobox(frame, values=list(self.taxas_gmd.keys()), state="readonly")
                ent.set("Pasto")
            else:
                ent = ttk.Entry(frame)
            ent.grid(row=r * 2 + 1, column=c, sticky="ew", padx=5, pady=2)
            self.inputs[key] = ent

        foto_row = ttk.Frame(self.aba1)
        foto_row.pack(fill="x", padx=10)
        ttk.Button(foto_row, text="📸 Selecionar Foto", command=self.selecionar_foto).pack(side="left", padx=5)
        self.lbl_foto = ttk.Label(foto_row, text="Nenhuma foto selecionada")
        self.lbl_foto.pack(side="left", padx=8)

        self.preview_label = ttk.Label(self.aba1, text="Preview indisponível")
        self.preview_label.pack(anchor="w", padx=15, pady=3)

        btns = ttk.Frame(self.aba1)
        btns.pack(pady=5)
        ttk.Button(btns, text="💾 Salvar Animal", command=self.db_salvar_animal).pack(side="left", padx=5)
        ttk.Button(btns, text="✅ Marcar como Vendido", command=self.marcar_como_vendido).pack(side="left", padx=5)
        ttk.Button(btns, text="🗑️ Excluir", command=self.db_excluir_animal).pack(side="left", padx=5)
        ttk.Button(btns, text="✨ Limpar Campos", command=self.limpar_campos_animal).pack(side="left", padx=5)

        self.tree = ttk.Treeview(self.aba1, columns=("ID", "N", "Peso Atual", "Arrobas", "Investimento", "Lucro Estimado"), show="headings")
        for col in self.tree["columns"]:
            self.tree.heading(col, text=col)
        self.tree.pack(fill="both", expand=True, padx=10, pady=5)
        self.tree.bind("<<TreeviewSelect>>", self.carregar_campos_selecionados)

    def setup_aba_vacinas(self):
        frame_v = ttk.LabelFrame(self.aba2, text="Gestão de Vacinação Completa")
        frame_v.pack(fill="x", padx=10, pady=10)

        ttk.Label(frame_v, text="ID Animal:").grid(row=0, column=0, padx=5, sticky="w")
        self.vac_id_animal = ttk.Entry(frame_v, width=12)
        self.vac_id_animal.grid(row=0, column=1, padx=5)

        ttk.Label(frame_v, text="Nome da Vacina/Marca:").grid(row=0, column=2, padx=5, sticky="w")
        self.vac_nome = ttk.Entry(frame_v, width=25)
        self.vac_nome.grid(row=0, column=3, padx=5)

        ttk.Label(frame_v, text="Tipo de Vacina:").grid(row=0, column=4, padx=5, sticky="w")
        self.vac_tipo = ttk.Combobox(frame_v, values=self.tipos_vacina, state="readonly", width=15)
        self.vac_tipo.set("Aftosa")
        self.vac_tipo.grid(row=0, column=5, padx=5)

        ttk.Label(frame_v, text="Data de Aplicação:").grid(row=1, column=0, padx=5, pady=10, sticky="w")
        self.vac_dt_aplicacao = ttk.Entry(frame_v, width=12)
        self.vac_dt_aplicacao.insert(0, datetime.now().strftime(DATE_FMT))
        self.vac_dt_aplicacao.grid(row=1, column=1, padx=5)

        ttk.Label(frame_v, text="Data de Vencimento:").grid(row=1, column=2, padx=5, sticky="w")
        self.vac_dt_vencimento = ttk.Entry(frame_v, width=12)
        self.vac_dt_vencimento.grid(row=1, column=3, padx=5)

        btns_v = ttk.Frame(frame_v)
        btns_v.grid(row=1, column=4, columnspan=2, sticky="e", padx=5)
        ttk.Button(btns_v, text="💾 Salvar", command=self.db_salvar_vacina).pack(side="left", padx=2)
        ttk.Button(btns_v, text="🗑️ Excluir", command=self.db_excluir_vacina).pack(side="left", padx=2)
        ttk.Button(btns_v, text="✨ Limpar", command=self.limpar_campos_vacina).pack(side="left", padx=2)

        search_box = ttk.LabelFrame(self.aba2, text="Pesquisar Vacinas e Histórico Sanitário")
        search_box.pack(fill="x", padx=10, pady=5)
        ttk.Label(search_box, text="ID/N° Animal ou Vacina:").pack(side="left", padx=5)
        self.hist_busca_vac = ttk.Entry(search_box, width=20)
        self.hist_busca_vac.pack(side="left", padx=5)
        ttk.Button(search_box, text="🔎 Filtrar Vacinas", command=self.filtrar_vacinas).pack(side="left", padx=5)
        ttk.Button(search_box, text="🔎 Buscar Histórico", command=self.buscar_historico_vacina).pack(side="left", padx=5)
        ttk.Button(search_box, text="🔁 Limpar Filtro", command=self.limpar_pesquisa_vacina).pack(side="left", padx=5)

        self.tree_vac = ttk.Treeview(self.aba2, columns=("ID", "ID Animal", "Vacina/Marca", "Tipo", "Apli.", "Venc.", "Status"), show="headings")
        self.tree_vac.heading("ID", text="Ref.")
        self.tree_vac.heading("ID Animal", text="Animal (ID)")
        self.tree_vac.heading("Vacina/Marca", text="Vacina / Marca")
        self.tree_vac.heading("Tipo", text="Categoria")
        self.tree_vac.heading("Apli.", text="Aplicação")
        self.tree_vac.heading("Venc.", text="Vencimento")
        self.tree_vac.heading("Status", text="Status Sanitário")
        self.tree_vac.column("ID", width=40)
        self.tree_vac.column("Status", width=120)
        self.tree_vac.pack(fill="both", expand=True, padx=10, pady=5)
        self.tree_vac.bind("<<TreeviewSelect>>", self.carregar_vacina_selecionada)

        self.tree_hist_vac = ttk.Treeview(self.aba2, columns=("ID", "Data", "Evento", "Origem", "Descrição"), show="headings", height=6)
        for c in self.tree_hist_vac["columns"]:
            self.tree_hist_vac.heading(c, text=c)
        self.tree_hist_vac.pack(fill="x", padx=10, pady=(0, 10))

        ttk.Button(self.aba2, text="📄 Gerar Relatório PDF", command=self.gerar_relatorio_pdf).pack(pady=(0, 10))

    def setup_aba_dashboard(self):
        self.dash_container = ttk.Frame(self.aba3)
        self.dash_container.pack(fill="both", expand=True)
        ttk.Button(self.dash_container, text="🔄 Recarregar Dashboard", command=self.gerar_dashboard).pack(pady=10)
        self.fig_frame = ttk.Frame(self.dash_container)
        self.fig_frame.pack(fill="both", expand=True)

    def setup_aba_vendidos(self):
        top = ttk.Frame(self.aba4)
        top.pack(fill="x", padx=10, pady=8)
        ttk.Label(top, text="Animais marcados como vendidos (dados preservados no banco)").pack(side="left")
        ttk.Button(top, text="🔄 Atualizar", command=self.atualizar_tabela_vendidos).pack(side="right", padx=5)

        self.tree_vendidos = ttk.Treeview(
            self.aba4,
            columns=("ID", "N°", "Raça", "Data Compra", "Data Venda", "Valor Compra", "@ Venda"),
            show="headings",
        )
        for c in self.tree_vendidos["columns"]:
            self.tree_vendidos.heading(c, text=c)
        self.tree_vendidos.pack(fill="both", expand=True, padx=10, pady=5)

        ttk.Button(self.aba4, text="↩ Reativar Animal Selecionado", command=self.reativar_animal_vendido).pack(pady=(0, 10))

    def selecionar_foto(self):
        path = filedialog.askopenfilename(filetypes=[("Imagens", "*.png;*.jpg;*.jpeg;*.gif")])
        if not path:
            return
        self.foto_path_atual = path
        self.lbl_foto.configure(text=os.path.basename(path))
        self.atualizar_preview_foto(path)

    def atualizar_preview_foto(self, path):
        if not path or not os.path.exists(path):
            self.preview_label.configure(image="", text="Preview indisponível")
            self._foto_preview_ref = None
            return

        if Image is None or ImageTk is None:
            self.preview_label.configure(image="", text=f"Foto: {os.path.basename(path)}")
            self._foto_preview_ref = None
            return

        try:
            img = Image.open(path)
            img.thumbnail((160, 160))
            tk_img = ImageTk.PhotoImage(img)
            self.preview_label.configure(image=tk_img, text="")
            self._foto_preview_ref = tk_img
        except Exception:
            self.preview_label.configure(image="", text=f"Foto: {os.path.basename(path)}")
            self._foto_preview_ref = None

    def calcular_performance(self, row):
        try:
            peso_inicial = float(row["peso_ini"])

            # Regra solicitada: só calcular ganho de peso quando houver data de venda.
            if row.get("dt_vend"):
                dt_com = datetime.strptime(row["dt_com"], DATE_FMT)
                dt_fim = datetime.strptime(row["dt_vend"], DATE_FMT)
                dias = max((dt_fim - dt_com).days, 0)
                peso = peso_inicial + (dias * self.taxas_gmd.get(row["tipo_confi"], 0))
            else:
                peso = peso_inicial

            arrobas = peso / 30.0
            lucro = (arrobas * float(row["val_arr_venda"])) - float(row["val_com"])
            return peso, arrobas, lucro
        except (ValueError, TypeError, KeyError):
            return 0.0, 0.0, 0.0

    def atualizar_tabela(self):
        for i in self.tree.get_children():
            self.tree.delete(i)
        with self.db.get_conn() as conn:
            df = pd.read_sql_query("SELECT * FROM animais WHERE status='ATIVO'", conn)

        for _, row in df.iterrows():
            peso, arroba, lucro = self.calcular_performance(row)
            self.tree.insert("", "end", values=(
                row["id"], row["n_animal"], f"{peso:.1f} kg", f"{arroba:.2f} @",
                self.format_moeda(row["val_com"]), self.format_moeda(lucro)
            ))
        self.atualizar_tabela_vendidos()

    def validar_animal(self, d):
        required = ["n_animal", "raca", "dt_com", "tipo_confi"]
        faltantes = [k for k in required if not d.get(k)]
        if faltantes:
            messagebox.showwarning("Validação", f"Preencha os campos obrigatórios: {', '.join(faltantes)}")
            return False
        if not self.parse_date(d.get("dt_com"), required=True):
            messagebox.showwarning("Validação", "Data de compra inválida. Use DD/MM/AAAA")
            return False
        if d.get("dt_vend") and not self.parse_date(d.get("dt_vend"), required=False):
            messagebox.showwarning("Validação", "Data de venda inválida. Use DD/MM/AAAA")
            return False
        return True

    def db_salvar_animal(self):
        d = {k: v.get().strip() for k, v in self.inputs.items()}
        if not self.validar_animal(d):
            return

        try:
            with self.db.get_conn() as conn:
                if self.selected_id is None:
                    cur = conn.execute(
                        """INSERT INTO animais
                        (n_animal, raca, dt_com, val_com, peso_ini, tipo_confi, val_arr_venda, dt_vend, status, cor, desc_carac, foto_path)
                        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (d['n_animal'], d['raca'], d['dt_com'], self.parse_float(d['val_com']),
                         self.parse_float(d['peso_ini']), d['tipo_confi'],
                         self.parse_float(d['val_arr_venda']), d['dt_vend'], 'ATIVO', d['cor'], d['desc_carac'], self.foto_path_atual)
                    )
                    self.registrar_evento(conn, cur.lastrowid, "Cadastro", "Animal cadastrado.", "animal")
                else:
                    conn.execute(
                        """UPDATE animais SET
                        n_animal=?, raca=?, dt_com=?, val_com=?, peso_ini=?,
                        tipo_confi=?, val_arr_venda=?, dt_vend=?, status='ATIVO', cor=?, desc_carac=?, foto_path=? WHERE id=?""",
                        (d['n_animal'], d['raca'], d['dt_com'], self.parse_float(d['val_com']),
                         self.parse_float(d['peso_ini']), d['tipo_confi'],
                         self.parse_float(d['val_arr_venda']), d['dt_vend'], d['cor'], d['desc_carac'], self.foto_path_atual, self.selected_id)
                    )
                    self.registrar_evento(conn, int(self.selected_id), "Atualização", "Cadastro do animal atualizado.", "animal")

            self.atualizar_tabela()
            self.limpar_campos_animal()
            self.gerar_dashboard()
            messagebox.showinfo("Sucesso", "Animal salvo com sucesso!")
        except Exception as e:
            messagebox.showerror("Erro de Cadastro", str(e))

    def marcar_como_vendido(self):
        if not self.selected_id:
            messagebox.showwarning("Venda", "Selecione um animal para marcar como vendido.")
            return

        data_venda = self.inputs["dt_vend"].get().strip() or datetime.now().strftime(DATE_FMT)
        if not self.parse_date(data_venda, required=True):
            messagebox.showwarning("Venda", "Data de venda inválida. Use DD/MM/AAAA")
            return

        if not messagebox.askyesno("Confirmar Venda", f"Marcar animal {self.selected_id} como vendido?"):
            return

        try:
            with self.db.get_conn() as conn:
                conn.execute(
                    "UPDATE animais SET status='VENDIDO', dt_vend=? WHERE id=?",
                    (data_venda, self.selected_id),
                )
                self.registrar_evento(conn, int(self.selected_id), "Venda", f"Animal marcado como vendido em {data_venda}.", "animal")

            self.atualizar_tabela()
            self.atualizar_tabela_vacinas()
            self.gerar_dashboard()
            self.limpar_campos_animal()
            messagebox.showinfo("Venda", "Animal movido para a aba de vendidos com sucesso.")
        except Exception as e:
            messagebox.showerror("Venda", f"Falha ao marcar venda: {e}")

    def atualizar_tabela_vendidos(self):
        if not hasattr(self, "tree_vendidos"):
            return
        for i in self.tree_vendidos.get_children():
            self.tree_vendidos.delete(i)

        with self.db.get_conn() as conn:
            rows = conn.execute(
                """
                SELECT id, n_animal, raca, dt_com, dt_vend, val_com, val_arr_venda
                FROM animais
                WHERE status='VENDIDO'
                ORDER BY id DESC
                """
            ).fetchall()

        for row in rows:
            self.tree_vendidos.insert("", "end", values=row)

    def reativar_animal_vendido(self):
        sel = self.tree_vendidos.selection()
        if not sel:
            messagebox.showwarning("Reativar", "Selecione um animal vendido para reativar.")
            return
        item = self.tree_vendidos.item(sel[0])["values"]
        animal_id = int(item[0])

        if not messagebox.askyesno("Reativar", f"Reativar animal {animal_id} para status ATIVO?"):
            return

        with self.db.get_conn() as conn:
            conn.execute("UPDATE animais SET status='ATIVO', dt_vend='' WHERE id=?", (animal_id,))
            self.registrar_evento(conn, animal_id, "Reativação", "Animal reativado para gestão ativa.", "animal")

        self.atualizar_tabela()
        self.atualizar_tabela_vacinas()
        self.gerar_dashboard()
        messagebox.showinfo("Reativar", "Animal reativado com sucesso.")

    def carregar_campos_selecionados(self, _):
        sel = self.tree.selection()
        if not sel:
            return
        self.selected_id = self.tree.item(sel[0])['values'][0]

        with self.db.get_conn() as conn:
            res = conn.execute("SELECT id, n_animal, raca, cor, dt_com, val_com, peso_ini, tipo_confi, val_arr_venda, dt_vend, desc_carac, foto_path FROM animais WHERE id=?", (self.selected_id,)).fetchone()
        if not res:
            return
        keys = ["id", "n_animal", "raca", "cor", "dt_com", "val_com", "peso_ini", "tipo_confi", "val_arr_venda", "dt_vend", "desc_carac", "foto_path"]
        row = dict(zip(keys, res))

        for k in self.inputs:
            self.inputs[k].delete(0, tk.END)
            val = row.get(k)
            if isinstance(val, float):
                self.inputs[k].insert(0, f"{val:.2f}".replace(".", ","))
            else:
                self.inputs[k].insert(0, str(val) if val else "")

        self.foto_path_atual = row.get("foto_path")
        self.lbl_foto.configure(text=os.path.basename(self.foto_path_atual) if self.foto_path_atual else "Nenhuma foto selecionada")
        self.atualizar_preview_foto(self.foto_path_atual)

        self.vac_id_animal.delete(0, tk.END)
        self.vac_id_animal.insert(0, str(self.selected_id))

    def db_excluir_animal(self):
        if not self.selected_id:
            return
        if messagebox.askyesno("Excluir", "Excluir este animal e suas vacinas?"):
            with self.db.get_conn() as conn:
                self.registrar_evento(conn, int(self.selected_id), "Exclusão", "Animal removido.", "animal")
                conn.execute("DELETE FROM animais WHERE id=?", (self.selected_id,))
            self.atualizar_tabela()
            self.atualizar_tabela_vacinas()
            self.limpar_campos_animal()
            self.gerar_dashboard()

    def limpar_campos_animal(self):
        self.selected_id = None
        self.foto_path_atual = None
        self.lbl_foto.configure(text="Nenhuma foto selecionada")
        self.atualizar_preview_foto(None)
        for v in self.inputs.values():
            if isinstance(v, ttk.Combobox):
                v.set("Pasto")
            else:
                v.delete(0, tk.END)

    def calcular_status_vacina(self, dt_venc_str):
        if not dt_venc_str:
            return "⚪ S/ Venc."
        try:
            venc = datetime.strptime(dt_venc_str, DATE_FMT)
            if venc >= datetime.now():
                return "🟢 OK"
            return "🔴 NOK (Vencida)"
        except ValueError:
            return "⚠️ Data Inválida"

    def atualizar_tabela_vacinas(self):
        for i in self.tree_vac.get_children():
            self.tree_vac.delete(i)
        with self.db.get_conn() as conn:
            res = conn.execute(
                """
                SELECT v.id, a.id, v.nome_vacina, v.tipo_vacina, v.dt_aplicacao, v.dt_vencimento
                FROM vacinas v JOIN animais a ON v.id_animal = a.id
                ORDER BY v.id DESC
                """
            ).fetchall()

        for r in res:
            id_registro, id_animal, nome, tipo, dt_apl, dt_venc = r
            status = self.calcular_status_vacina(dt_venc)
            self.tree_vac.insert("", "end", values=(id_registro, id_animal, nome, tipo, dt_apl, dt_venc, status))

    def filtrar_vacinas(self):
        termo = self.hist_busca_vac.get().strip()

        for i in self.tree_vac.get_children():
            self.tree_vac.delete(i)

        with self.db.get_conn() as conn:
            if not termo:
                res = conn.execute(
                    """
                    SELECT v.id, a.id, v.nome_vacina, v.tipo_vacina, v.dt_aplicacao, v.dt_vencimento
                    FROM vacinas v
                    JOIN animais a ON v.id_animal = a.id
                    ORDER BY v.id DESC
                    """
                ).fetchall()
            else:
                like = f"%{termo}%"
                res = conn.execute(
                    """
                    SELECT v.id, a.id, v.nome_vacina, v.tipo_vacina, v.dt_aplicacao, v.dt_vencimento
                    FROM vacinas v
                    JOIN animais a ON v.id_animal = a.id
                    WHERE CAST(a.id AS TEXT) = ?
                       OR a.n_animal LIKE ?
                       OR v.nome_vacina LIKE ?
                       OR v.tipo_vacina LIKE ?
                    ORDER BY v.id DESC
                    """,
                    (termo, like, like, like),
                ).fetchall()

        for r in res:
            id_registro, id_animal, nome, tipo, dt_apl, dt_venc = r
            status = self.calcular_status_vacina(dt_venc)
            self.tree_vac.insert("", "end", values=(id_registro, id_animal, nome, tipo, dt_apl, dt_venc, status))

        if not res:
            messagebox.showinfo("Pesquisa", "Nenhuma vacina encontrada para o filtro informado.")

    def limpar_pesquisa_vacina(self):
        self.hist_busca_vac.delete(0, tk.END)
        for i in self.tree_hist_vac.get_children():
            self.tree_hist_vac.delete(i)
        self.atualizar_tabela_vacinas()

    def carregar_vacina_selecionada(self, _):
        sel = self.tree_vac.selection()
        if not sel:
            return
        valores = self.tree_vac.item(sel[0])['values']

        self.vac_selected_id = int(valores[0])
        self.vac_id_animal.delete(0, tk.END)
        self.vac_id_animal.insert(0, str(int(valores[1])))
        self.vac_nome.delete(0, tk.END)
        self.vac_nome.insert(0, str(valores[2]))
        self.vac_tipo.set(str(valores[3]))
        self.vac_dt_aplicacao.delete(0, tk.END)
        self.vac_dt_aplicacao.insert(0, str(valores[4]))
        self.vac_dt_vencimento.delete(0, tk.END)
        self.vac_dt_vencimento.insert(0, str(valores[5]))

        self.hist_busca_vac.delete(0, tk.END)
        self.hist_busca_vac.insert(0, str(int(valores[1])))
        self.buscar_historico_vacina()

    def buscar_historico_vacina(self):
        for i in self.tree_hist_vac.get_children():
            self.tree_hist_vac.delete(i)

        chave = self.hist_busca_vac.get().strip()
        if not chave:
            return

        with self.db.get_conn() as conn:
            like = f"%{chave}%"
            rows = conn.execute(
                """
                SELECT h.id, h.data, h.tipo_evento, h.origem, h.descricao
                FROM historico h
                JOIN animais a ON a.id = h.id_animal
                WHERE CAST(a.id AS TEXT) = ?
                   OR a.n_animal LIKE ?
                   OR h.descricao LIKE ?
                ORDER BY h.id DESC
                LIMIT 200
                """,
                (chave, like, like),
            ).fetchall()

        for row in rows:
            self.tree_hist_vac.insert("", "end", values=row)

        if not rows:
            messagebox.showinfo("Histórico", "Nenhum histórico encontrado para a busca informada.")

    def db_salvar_vacina(self):
        idx = self.vac_id_animal.get().strip()
        nome = self.vac_nome.get().strip()
        tipo = self.vac_tipo.get().strip()
        dt_apl = self.vac_dt_aplicacao.get().strip()
        dt_venc = self.vac_dt_vencimento.get().strip()

        if not idx or not nome or not dt_venc:
            messagebox.showwarning("Aviso", "Preencha ID, Nome da Vacina e Data de Vencimento!")
            return

        if dt_apl and not self.parse_date(dt_apl, required=False):
            messagebox.showwarning("Validação", "Data de aplicação inválida. Use DD/MM/AAAA")
            return

        if not self.parse_date(dt_venc, required=True):
            messagebox.showwarning("Validação", "Data de vencimento inválida. Use DD/MM/AAAA")
            return

        try:
            with self.db.get_conn() as conn:
                animal = conn.execute("SELECT id FROM animais WHERE id=?", (idx,)).fetchone()
                if not animal:
                    return messagebox.showerror("Erro", f"Animal ID {idx} não existe.")

                if self.vac_selected_id is None:
                    conn.execute(
                        "INSERT INTO vacinas (id_animal, nome_vacina, tipo_vacina, dt_aplicacao, dt_vencimento) VALUES (?,?,?,?,?)",
                        (idx, nome, tipo, dt_apl, dt_venc)
                    )
                    self.registrar_evento(conn, int(idx), "Vacinação", f"Aplicada: {nome} ({tipo}) venc. {dt_venc}", "vacina")
                else:
                    conn.execute(
                        "UPDATE vacinas SET id_animal=?, nome_vacina=?, tipo_vacina=?, dt_aplicacao=?, dt_vencimento=? WHERE id=?",
                        (idx, nome, tipo, dt_apl, dt_venc, self.vac_selected_id)
                    )
                    self.registrar_evento(conn, int(idx), "Vacinação", f"Vacina atualizada: {nome} ({tipo})", "vacina")

            self.limpar_campos_vacina()
            self.atualizar_tabela_vacinas()
            self.gerar_dashboard()
            self.hist_busca_vac.delete(0, tk.END)
            self.hist_busca_vac.insert(0, idx)
            self.buscar_historico_vacina()
            messagebox.showinfo("Sucesso", "Vacina salva com sucesso!")
        except Exception as e:
            messagebox.showerror("Erro", str(e))

    def db_excluir_vacina(self):
        if not self.vac_selected_id:
            return
        idx = self.vac_id_animal.get().strip()
        nome = self.vac_nome.get().strip()
        if messagebox.askyesno("Excluir", "Deseja excluir esta vacina?"):
            with self.db.get_conn() as conn:
                conn.execute("DELETE FROM vacinas WHERE id=?", (self.vac_selected_id,))
                if idx:
                    self.registrar_evento(conn, int(idx), "Vacinação", f"Vacina removida: {nome}", "vacina")
            self.limpar_campos_vacina()
            self.atualizar_tabela_vacinas()
            self.gerar_dashboard()
            if idx:
                self.hist_busca_vac.delete(0, tk.END)
                self.hist_busca_vac.insert(0, idx)
                self.buscar_historico_vacina()

    def limpar_campos_vacina(self):
        self.vac_selected_id = None
        self.vac_id_animal.delete(0, tk.END)
        self.vac_nome.delete(0, tk.END)
        self.vac_tipo.set("Aftosa")
        self.vac_dt_aplicacao.delete(0, tk.END)
        self.vac_dt_aplicacao.insert(0, datetime.now().strftime(DATE_FMT))
        self.vac_dt_vencimento.delete(0, tk.END)

    def gerar_relatorio_pdf(self):
        file_path = filedialog.asksaveasfilename(
            title="Salvar relatório em PDF",
            defaultextension=".pdf",
            filetypes=[("PDF", "*.pdf")],
            initialfile=f"relatorio_pecuaria_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
        )
        if not file_path:
            return

        with self.db.get_conn() as conn:
            df_animais = pd.read_sql_query("SELECT * FROM animais ORDER BY id", conn)
            df_ativos = pd.read_sql_query("SELECT * FROM animais WHERE status='ATIVO' ORDER BY id", conn)
            df_vendidos = pd.read_sql_query("SELECT * FROM animais WHERE status='VENDIDO' ORDER BY id", conn)
            df_vac = pd.read_sql_query(
                """
                SELECT v.id, v.id_animal, v.nome_vacina, v.tipo_vacina, v.dt_aplicacao, v.dt_vencimento
                FROM vacinas v
                ORDER BY v.id DESC
                """,
                conn,
            )

        if df_animais.empty and df_vac.empty:
            messagebox.showwarning("Relatório", "Não há dados para gerar o relatório.")
            return

        if not df_vac.empty:
            df_vac["status"] = df_vac["dt_vencimento"].apply(self.calcular_status_vacina)

        try:
            with PdfPages(file_path) as pdf:
                fig1, ax1 = plt.subplots(figsize=(8.27, 11.69))
                ax1.axis("off")
                linhas = [
                    "RELATÓRIO PECUÁRIA",
                    f"Gerado em: {datetime.now().strftime('%d/%m/%Y %H:%M')}",
                    "",
                    f"Total de animais (geral): {len(df_animais)}",
                    f"Animais ativos: {len(df_ativos)}",
                    f"Animais vendidos: {len(df_vendidos)}",
                    f"Total de vacinas: {len(df_vac)}",
                ]
                if not df_animais.empty:
                    total_invest = float(df_animais["val_com"].fillna(0).sum())
                    linhas.append(f"Investimento total: {self.format_moeda(total_invest)}")
                ax1.text(0.05, 0.95, "\n".join(linhas), va="top", fontsize=12)
                pdf.savefig(fig1)
                plt.close(fig1)

                if not df_animais.empty:
                    fig2 = plt.Figure(figsize=(10, 4), dpi=100)
                    ax = fig2.add_subplot(111)
                    lucro_por_manejo = []
                    for _, row in df_animais.iterrows():
                        lucro_por_manejo.append(self.calcular_performance(row)[2])
                    df_tmp = df_animais.copy()
                    df_tmp["lucro"] = lucro_por_manejo
                    gp = df_tmp.groupby("tipo_confi")["lucro"].sum()
                    gp.plot(kind="bar", ax=ax, color=["#2ecc71", "#3498db", "#e67e22"])
                    ax.set_title("Lucro estimado por manejo (R$)")
                    ax.set_ylabel("Reais")
                    fig2.tight_layout()
                    pdf.savefig(fig2)
                    plt.close(fig2)

                if not df_vac.empty:
                    fig3 = plt.Figure(figsize=(10, 5), dpi=100)
                    ax = fig3.add_subplot(111)
                    df_vac["tipo_vacina"].value_counts().plot(kind="barh", ax=ax, color="#9b59b6")
                    ax.set_title("Vacinas por tipo")
                    fig3.tight_layout()
                    pdf.savefig(fig3)
                    plt.close(fig3)

            messagebox.showinfo("Relatório", f"Relatório PDF gerado com sucesso:\n{file_path}")
        except Exception as e:
            messagebox.showerror("Relatório", f"Falha ao gerar PDF: {e}")

    def gerar_dashboard(self):
        for w in self.fig_frame.winfo_children():
            w.destroy()

        with self.db.get_conn() as conn:
            df_animais = pd.read_sql_query("SELECT * FROM animais WHERE status='ATIVO'", conn)
            df_vac = pd.read_sql_query("SELECT tipo_vacina, dt_vencimento FROM vacinas", conn)

        if df_animais.empty and df_vac.empty:
            ttk.Label(self.fig_frame, text="Nenhum dado para exibir.").pack(pady=20)
            return

        fig = plt.Figure(figsize=(12, 4), dpi=100)

        if not df_animais.empty:
            ax1 = fig.add_subplot(131)
            df_animais['Lucro'] = df_animais.apply(lambda r: self.calcular_performance(r)[2], axis=1)
            df_animais.groupby('tipo_confi')['Lucro'].sum().plot(kind='bar', ax=ax1, color=['#2ecc71', '#3498db', '#e67e22'])
            ax1.set_title("Lucro Estimado (R$)")
            ax1.set_ylabel("Reais")

        if not df_vac.empty:
            ax2 = fig.add_subplot(132)
            df_vac['tipo_vacina'].value_counts().plot(kind='barh', ax=ax2, color='#9b59b6')
            ax2.set_title("Volume por Tipo de Vacina")

            ax3 = fig.add_subplot(133)
            df_vac['Status'] = df_vac['dt_vencimento'].apply(lambda x: "OK" if "🟢" in self.calcular_status_vacina(x) else "NOK")
            contagem_status = df_vac['Status'].value_counts()
            cores = ['#2ecc71' if status == 'OK' else '#e74c3c' for status in contagem_status.index]
            ax3.pie(contagem_status, labels=contagem_status.index, autopct='%1.1f%%', colors=cores, startangle=90)
            ax3.set_title("Status Sanitário do Rebanho")

        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.fig_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)


if __name__ == "__main__":
    root = tk.Tk()
    if sv_ttk:
        sv_ttk.set_theme("dark")
    app = AppPecuariaCRUD(root)
    root.mainloop()
