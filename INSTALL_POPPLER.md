# Installation de Poppler pour Windows

## Méthode 1: Installation rapide (Recommandée)

### 1. Télécharger Poppler

Téléchargez la dernière version depuis: https://github.com/oschwartz10612/poppler-windows/releases/

Prenez: `Release-XX.XX.X-0.zip`

### 2. Extraire

Extrayez le ZIP dans un dossier, par exemple:
```
C:\poppler-XX.XX.X\
```

### 3. Ajouter au PATH

**Option A: Via l'interface Windows**

1. Clic droit sur "Ce PC" → Propriétés
2. Paramètres système avancés
3. Variables d'environnement
4. Dans "Variables système", sélectionner "Path" → Modifier
5. Nouveau → Ajouter: `C:\poppler-XX.XX.X\Library\bin`
6. OK partout
7. **Redémarrer le terminal**

**Option B: Via PowerShell (Admin)**

```powershell
$env:Path += ";C:\poppler-XX.XX.X\Library\bin"
[Environment]::SetEnvironmentVariable("Path", $env:Path, [EnvironmentVariableTarget]::Machine)
```

### 4. Vérifier l'installation

```bash
pdftoppm -v
```

Vous devriez voir la version de Poppler.

---

## Méthode 2: Via Chocolatey (si installé)

```bash
choco install poppler
```

---

## Méthode 3: Via Conda (si vous utilisez Conda)

```bash
conda install -c conda-forge poppler
```

---

## Test

Après installation, testez avec:

```bash
python -c "from pdf2image import convert_from_path; print('OK')"
```

Si pas d'erreur → C'est bon!

Relancez ensuite:
```bash
python generate.py
```

Et sélectionnez l'option avec PDFs scannés.

---

## Alternative: Désactiver l'OCR temporairement

Si vous n'avez pas besoin de traiter les PDFs scannés maintenant:

Dans le menu de `generate.py`, choisissez:
```
Type de PDFs à traiter:
  1. Tous (texte natif + scans)
  2. Uniquement texte natif  ← Choisir ceci
  3. Uniquement scans
```

Ainsi, seuls les PDFs natifs seront traités (pas besoin de Poppler).
