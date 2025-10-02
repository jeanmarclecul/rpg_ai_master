#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rpg_ai_master.py
Jeu de rôle papier géré par un modèle IA local (Hugging Face transformers).

Usage:
    python rpg_ai_master.py --config config.json

Le modèle (paramètre dans config.json) est chargé localement via transformers.
Le script orchestre :
 - le Maître de Jeu (GM) géré par le modèle,
 - plusieurs joueurs "IA" (pris en charge par le même modèle),
 - et des joueurs "humains" (saisie stdin).
Le modèle s'occupe également de créer/tenir à jour les fiches de perso si demandé
(la création est demandée au modèle, le code n'essaie pas de générer les fiches lui-même).
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Dict

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
    import torch
except Exception as e:
    print("Erreur d'import : installez les dépendances via install.sh (venv).")
    raise

# ---------- Utilitaires ----------
def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def safe_print(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()

# ---------- Génération via Transformers ----------
class LocalModel:
    def __init__(self, model_name: str, device: int = None, generation_kwargs: dict = None):
        """
        device: None => auto (cuda if available), else -1 for cpu
        """
        self.model_name = model_name
        self.generation_kwargs = generation_kwargs or {}
        # determine device
        if device is None:
            if torch.cuda.is_available():
                self.device = 0
            else:
                self.device = -1
        else:
            self.device = device

        safe_print(f"[MODEL] Chargement du tokenizer et du modèle '{model_name}' (device={self.device})... (cela peut prendre du temps)")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        # try causal LM then seq2seq
        model = None
        try:
            model = AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True)
            self.task = "text-generation"
        except Exception:
            try:
                model = AutoModelForSeq2SeqLM.from_pretrained(model_name, low_cpu_mem_usage=True)
                self.task = "text2text-generation"
            except Exception as e:
                raise RuntimeError(f"Impossible de charger le modèle '{model_name}': {e}")

        # pipeline with tokenizer and model
        self.pipe = pipeline(self.task, model=model, tokenizer=self.tokenizer, device=self.device if self.device != -1 else 'cpu')

    def generate(self, prompt: str, **override_kwargs) -> str:
        kwargs = dict(self.generation_kwargs)
        kwargs.update(override_kwargs)
        # simple generation; pipeline returns list of dicts
        out = self.pipe(prompt, **kwargs)
        # depending on pipeline version it returns a list with 'generated_text' or 'text'
        text = out[0].get("generated_text") or out[0].get("text") or str(out[0])
        return text

# ---------- Jeu ----------
class RPGGame:
    def __init__(self, cfg: Dict):
        self.cfg = cfg
        # formatting placeholders
        self.nb_players = int(cfg.get("nb_players", 4))
        self.nb_ai_players = int(cfg.get("nb_ai_players", 2))
        self.nb_human_players = max(0, self.nb_players - self.nb_ai_players)

        model_name = cfg.get("model_name", "gpt2-xl")
        gen_kwargs = cfg.get("generation", {
            "max_new_tokens": 400,
            "temperature": 0.9,
            "top_p": 0.95,
            "do_sample": True,
        })
        # allow device override in config
        device = cfg.get("device", None)
        self.model = LocalModel(model_name, device=device, generation_kwargs=gen_kwargs)

        # conversation / story state
        self.history: List[Dict] = []  # list of messages {role: "system|gm|playerX|narration", content: str}
        self.turn = 0
        self.player_names = [f"J{idx+1}" for idx in range(self.nb_players)]
        # mark which players are IA
        self.ai_player_indexes = list(range(self.nb_ai_players))
        self.human_player_indexes = list(range(self.nb_ai_players, self.nb_players))

        # initial prompt template from config (user provided)
        # Template placeholders expected: {x} -> number of OTHER players? We'll map:
        # We'll set {x} = nb_ai_players (AI players count), {a} = nb_human_players, {z} = system de regles, {y} = style
        template = cfg.get("initial_prompt_template", "")
        # fill template safely
        self.initial_prompt = template.format(
            x=self.nb_ai_players,
            a=self.nb_human_players,
            z=cfg.get("system_of_rules", "règles standards"),
            y=cfg.get("style", "roleplay immersif")
        )

        # system instruction to model (set as first system message)
        sys_instr = cfg.get("system_message", f"Tu es le maître de jeu et un joueur IA pour une partie de jeu de rôle. Style: {cfg.get('style','')}")
        self.system_message = sys_instr

        # fiches: stored as dict of player index -> fiche text generated by model (but model generates content)
        self.fiches = {}

        # auto-create fiches?
        self.auto_create_fiches = bool(cfg.get("auto_create_fiches_for_humans", True))

        # persist filename for logs/fiches
        out_dir = Path(cfg.get("output_dir", "rpg_output"))
        out_dir.mkdir(parents=True, exist_ok=True)
        self.out_dir = out_dir

        # save cfg copy
        (out_dir / "used_config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")

    def save_state(self):
        s = {
            "turn": self.turn,
            "history": self.history,
            "fiches": self.fiches,
            "player_names": self.player_names
        }
        (self.out_dir / f"state_turn_{self.turn}.json").write_text(json.dumps(s, indent=2, ensure_ascii=False), encoding="utf-8")

    def request_model_create_fiche(self, player_idx: int, human: bool=True):
        """
        Ask the model to create a fiche de personnage for the given player index.
        We instruct the model to return JSON-like fiche. The model manages content.
        """
        player_name = self.player_names[player_idx]
        role = "humain" if human else "ia"
        prompt = (
            f"Tu vas **créer** la fiche de personnage du joueur '{player_name}' (type: {role}).\n"
            f"Système de règles: {self.cfg.get('system_of_rules')}\n"
            "Renvoie uniquement un JSON valide (objet) avec les champs essentiels suivants si pertinents:\n"
            "  - name, background, class, level, stats (object), compétences (list), équipement (list), short_description\n"
            "Important: tu es le maître de jeu IA, fais une fiche cohérente avec l'univers: "
            f"{self.cfg.get('universe', 'univers non spécifié')}.\n"
            "Ne fournis aucune explication hors du JSON. Si un champ n'est pas pertinent, mets null ou []."
            "\n\nJSON:"
        )
        # generate
        text = self.model.generate(prompt, max_new_tokens=400, temperature=0.8)
        # attempt to extract JSON part
        json_part = self._extract_json_like(text)
        if json_part:
            try:
                parsed = json.loads(json_part)
                self.fiches[player_name] = parsed
                safe_print(f"[FICHE CREATED] {player_name} -> saved.")
                (self.out_dir / f"fiche_{player_name}.json").write_text(json.dumps(parsed, indent=2, ensure_ascii=False), encoding="utf-8")
                return parsed
            except Exception:
                # fallback: store raw text
                self.fiches[player_name] = {"raw": text}
                (self.out_dir / f"fiche_{player_name}_raw.txt").write_text(text, encoding="utf-8")
                return self.fiches[player_name]
        else:
            self.fiches[player_name] = {"raw": text}
            (self.out_dir / f"fiche_{player_name}_raw.txt").write_text(text, encoding="utf-8")
            return self.fiches[player_name]

    def _extract_json_like(self, text: str) -> str:
        # naive extraction: find first '{' and last '}' and slice
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return text[start:end+1]
        return ""

    def start(self):
        safe_print("=== DÉMARRAGE DE LA PARTIE ===")
        safe_print(f"Nombre total joueurs: {self.nb_players} (IA: {self.nb_ai_players}, Humains: {self.nb_human_players})")
        safe_print("Initial prompt envoyé au modèle (formaté):")
        safe_print(self.initial_prompt)
        # place system message in history
        self.history.append({"role":"system", "content": self.system_message})
        # initial "setup" call to model: explain roles, ask to play GM + AI players
        setup_prompt = (
            f"{self.initial_prompt}\n\n"
            "Rappelle-toi: tu seras le Maître de Jeu (GM) et tu joueras également les personnages contrôlés par l'IA.\n"
            "Tu gères les actions, comptes et fiches des joueurs IA et tu aideras à créer/tenir les fiches des joueurs humains si demandé.\n"
            "Ne propose pas une liste de choix multi-choix fermés; décris la situation et prends des décisions pour les NPC/IA.\n"
            "Commence par présenter la scène d'ouverture et, si demandé, crée les fiches de personnages pour les joueurs humains automatiquement.\n"
        )
        # add to history and request generation
        self.history.append({"role":"assistant_setup", "content": setup_prompt})
        gm_intro = self.model.generate(setup_prompt, max_new_tokens=400, temperature=0.9)
        self.history.append({"role":"gm", "content": gm_intro})
        safe_print("\n[GM INTRO]\n")
        safe_print(gm_intro)
        # auto create fiches for humans if requested
        if self.auto_create_fiches and self.nb_human_players > 0:
            safe_print("\n[Création automatique des fiches pour joueurs humains demandée]")
            for idx in self.human_player_indexes:
                self.request_model_create_fiche(idx, human=True)

        # create fiches for IA players as well (model manages them)
        safe_print("\n[Création des fiches pour joueurs IA]")
        for idx in self.ai_player_indexes:
            self.request_model_create_fiche(idx, human=False)

        # main loop
        safe_print("\n=== Début du cycle de jeu (CTRL+C pour quitter) ===\n")
        try:
            while True:
                self.turn += 1
                safe_print(f"\n--- Tour {self.turn} ---")
                # GM describes scene and performs AI player actions
                prompt = self._build_gm_turn_prompt()
                gm_text = self.model.generate(prompt, max_new_tokens=600, temperature=0.95)
                self.history.append({"role":"gm", "content": gm_text})
                safe_print("\n[GM & IA actions]\n")
                safe_print(gm_text)

                # human players input
                if self.nb_human_players > 0:
                    for pidx in self.human_player_indexes:
                        pname = self.player_names[pidx]
                        safe_print(f"\nAction pour {pname} (joueur humain). Tape ton action (ou 'skip') :")
                        try:
                            action = input(f"{pname}> ").strip()
                        except KeyboardInterrupt:
                            safe_print("\nInterruption par l'utilisateur. Sauvegarde et sortie.")
                            self.save_state()
                            return
                        if action.lower() in ("quit","exit"):
                            safe_print("Fin de partie demandée. Sauvegarde et sortie.")
                            self.save_state()
                            return
                        if action == "":
                            action = "skip"
                        self.history.append({"role": f"player_{pname}", "content": action})

                # after human actions, ask model to resolve consequences
                resolve_prompt = self._build_resolution_prompt()
                resolution = self.model.generate(resolve_prompt, max_new_tokens=500, temperature=0.9)
                self.history.append({"role":"gm_resolution", "content": resolution})
                safe_print("\n[Résolution par le GM]\n")
                safe_print(resolution)

                # optional: ask model to update fiches if needed
                # We instruct the model to output fiche modifications in JSON if any change
                update_prompt = (
                    "Si une fiche de personnage doit changer à la suite des actions ci-dessus, "
                    "renvoie un objet JSON unique {'player': name, 'updates': { ... }} ou 'NONE' si aucune modification.\n"
                    "Ne renvoie rien d'autre.\n"
                )
                update_response = self.model.generate(update_prompt, max_new_tokens=200, temperature=0.6)
                # attempt parse
                if update_response.strip().upper().startswith("NONE"):
                    pass
                else:
                    js = self._extract_json_like(update_response)
                    if js:
                        try:
                            parsed = json.loads(js)
                            player = parsed.get("player")
                            updates = parsed.get("updates")
                            if player in self.fiches:
                                # merge shallow
                                if isinstance(self.fiches[player], dict):
                                    self.fiches[player].update(updates or {})
                                    (self.out_dir / f"fiche_{player}.json").write_text(json.dumps(self.fiches[player], indent=2, ensure_ascii=False), encoding="utf-8")
                                    safe_print(f"[Fiche mise à jour] {player}")
                        except Exception:
                            # ignore parse errors
                            pass

                # persist each N turns
                if self.turn % 2 == 0:
                    self.save_state()

                # small delay to avoid spin
                time.sleep(0.2)

        except KeyboardInterrupt:
            safe_print("\nArrêt demandé (KeyboardInterrupt). Sauvegarde de l'état.")
            self.save_state()
            safe_print("Terminé.")

    def _build_gm_turn_prompt(self) -> str:
        """
        Build a prompt that asks the model to narrate the next scene, perform IA players' actions,
        and advance the story. Provide relevant recent history.
        """
        # include last few history messages (to limit context)
        recent = [h for h in self.history[-8:]]  # last 8 entries
        recent_text = "\n\n".join([f"[{m['role'].upper()}]\n{m['content']}" for m in recent])
        prompt = (
            "Tu es le Maître de Jeu (GM). Utilise les informations ci-dessous (historique récent) "
            "pour décrire la suite de la scène, faire agir les personnages contrôlés par l'IA, "
            "et simuler les résultats des actions (sans demander de choix fermés au joueur humain).\n\n"
            f"{recent_text}\n\n"
            "Maintenant, décris la nouvelle situation, les actions des joueurs IA, et les conséquences immédiates.\n"
            "Sois précis, immersif, et respecte le système de règles: " + str(self.cfg.get("system_of_rules", "règles")) + ".\n"
        )
        return prompt

    def _build_resolution_prompt(self) -> str:
        """
        After human inputs, ask the model to resolve outcomes and continue narration.
        """
        recent = [h for h in self.history[-12:]]
        recent_text = "\n\n".join([f"[{m['role'].upper()}]\n{m['content']}" for m in recent])
        prompt = (
            "Tu es le Maître de Jeu. Résous maintenant les actions qui viennent d'être rapportées et "
            "décris les conséquences, l'évolution des PNJ, et propose la prochaine situation.\n\n"
            f"{recent_text}\n\n"
            "Rends une narration concise, puis détaille les conséquences mécaniques si approprié."
        )
        return prompt

# ---------- Entrée principale ----------
def main():
    parser = argparse.ArgumentParser(prog="rpg_ai_master", description="Jeu de rôle géré par modèle IA local")
    parser.add_argument("--config", "-c", type=str, default="config.json", help="Fichier JSON de configuration")
    args = parser.parse_args()

    cfg_path = args.config
    if not os.path.exists(cfg_path):
        safe_print(f"Fichier de configuration '{cfg_path}' introuvable.")
        sys.exit(1)

    cfg = load_config(cfg_path)
    game = RPGGame(cfg)
    game.start()

if __name__ == "__main__":
    main()
