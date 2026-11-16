# MogulCourseAnalyzer

## v1.8環境検証

2026-11-16の差分・バージョン・テスト結果・未確認事項は[ENVIRONMENT_V18.md](ENVIRONMENT_V18.md)を参照してください。
GUI検証のみを起動する場合：`.\.venv\Scripts\python.exe -m tools.v18_probe`。
これはアプリ本体ではありません。

Windows 11向けPythonデスクトップアプリの開発プロジェクトです。
現在はPhase 1の開発環境構築のみ完了対象で、アプリ本体・起動エントリーポイントは未実装です。

## 環境構築（Windows / PowerShell）

Python 3.13（64bit）、Git、PATHから実行できるFFmpegが必要です。
Pythonは[公式サイト](https://www.python.org/downloads/windows/)の署名付き64bitインストーラーでユーザー単位にインストールします。
Python導入後はPowerShellを開き直してください。

```powershell
Set-Location C:\work\MogulCourseAnalyzer
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

既存の`.venv`がある場合は作成コマンドを省略します。
仮想環境の有効化は任意です。上記のようにPythonを直接指定すれば、PowerShellの実行ポリシー変更は不要です。
`py`で見つからない場合は、次のPythonを直接使用できます。

```powershell
& "$env:LOCALAPPDATA\Programs\Python\Python313\python.exe" -m venv .venv
```

`requirements.in`は直接依存の一覧、`requirements.txt`は動作確認した推移的依存も含む固定バージョンの一覧です。
依存更新時はインストール・テスト後に次のコマンドで記録を更新します。

```powershell
.\.venv\Scripts\python.exe -m pip freeze | Set-Content -Encoding utf8 requirements.txt
```

## 動作確認

```powershell
git --version
ffmpeg -version
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pytest -v
```

テストではQtウィンドウを短時間表示してイベントループを確認し、OpenCV/NumPyの画像処理、Pandas/openpyxlのExcel往復、FFmpegの動画生成とOpenCVでの読み込みを確認します。
画面を利用できるWindowsデスクトップセッションで実行してください。
生成した画像・Excel・動画はpytestの一時ディレクトリに保存されます。

## アプリ起動方法

アプリ本体は未実装のため、現時点では起動コマンドはありません。
上記のpytestが環境の動作確認方法です。アプリ実装時に正式な起動方法を追記します。

## Gitと生成物

`.venv`、Pythonキャッシュ、テストキャッシュ、ビルド成果物、`outputs/`、`temp/`などは`.gitignore`で除外します。
動画などの大容量入力ファイルは現時点で追加せず、管理方針を決めてから配置してください。

## 初回構築の確認結果（2026-11-15）

- Windowsネイティブ x64 / PowerShell 7.6.9、OSビルド26300.9974（26H2）。レジストリのProductNameはWindows 10 Homeと表示されます。
- Python 3.13.12 / pip 26.3を導入し、`.venv`を作成しました。
- PySide6 6.12.0、OpenCV 5.0.0.93、NumPy 2.5.4、Pandas 3.1.1、openpyxl 3.1.5、pytest 9.2.0を導入しました。
- Git 2.54.0.windows.1、FFmpeg 4.1.3は既存環境を使用しました。
- `pip check`は問題なし、環境テストは4件すべて成功しました。
- Gitは`main`で初期化済み、コミットは未作成です。隔離実行ユーザーとの所有者差異に対応するため、ユーザーのGit設定の`safe.directory`に`C:/work/MogulCourseAnalyzer`のみ登録しました。
- FFmpegは古い既存版です。H.264生成・読み込みは確認済みですが、実際の入力動画の対応形式は今後の確認対象です。既存FFmpegはDropbox配下のPATHに依存しています。
