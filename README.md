# Mogul Course Analyzer v1.8

Windows 11 / Python / PySide6による日本語デスクトップアプリです。
Phase 1（映像入力とエリア設定）を実装しています。人物追跡・人数集計・イベント解析は後続フェーズです。

## 起動

既存の`.venv`をそのまま使います。削除・再作成や依存関係の更新は不要です。

```powershell
Set-Location C:\work\MogulCourseAnalyzer
.\.venv\Scripts\python.exe -m mogul
```

起動時に録画済み動画またはUVCを選択します。保存済み設定がある場合は前回の復元を選べます。

```powershell
.\.venv\Scripts\python.exe -m mogul --video "C:\path\course.mp4"
.\.venv\Scripts\python.exe -m mogul --database "C:\path\session.sqlite"
```

## 操作

1. 動画はファイル選択。UVCはデバイス・解像度・FPSを選びます。機器が指定モードに対応している必要があります。
2. AIモデル未導入のため自動検出不可と表示します。映像上をドラッグして初期四角形を作り、四隅のマーカーをドラッグしてミドルパートに合わせます。
3. 映像内のマーカー以外をシングルクリックすると、ダブルクリック判別の待ち時間後に確定し、マーカーが消えます。
4. 確定した領域内をダブルクリックすると再編集できます。ミドルパートを再編集するとレーン再設定が必要です。Undoで戻せます。
5. ミドルパート確定後、2〜4レーンを指定して「手動均等分割で初期配置」を押します。これはAI検出ではありません。
6. 各レーンを選択し、共有境界の上下マーカーで補正し、映像内クリックで個別に確定します。レーン選択状態で領域内をダブルクリックすると再編集します。
7. Ctrl+SでSQLiteへ保存、Ctrl+ZでUndo。ResetもUndoできます。ソース変更・終了時は未保存変更の保存確認が出ます。

最左レーンの左上・左下、最右レーンの右上・右下もドラッグできます。四隅はゲート位置やミドルパート外周に固定せず、ミドルパート内で上下左右に動かせます。実際のコブ列に合わせて幅を狭め、ゲート側との余白を設定してください。約2mという現場の目安を固定ピクセルへ換算することはありません。

内部境界は隣接レーンと共有し、変更時は両側が未確定に戻ります。外側の頂点変更は対象レーンのみ未確定に戻します。レーン間の空白は共有境界で防ぎ、重複・範囲外・自己交差を拒否します。レーンがミドルパート全体を埋める必要はありません。詳細は[レーン修正仕様](SPEC_V1_8.md)を参照してください。

レーン名・色、領域不透明度（初期30%）、ミドルパート色、左右反転を変更できます。反転後は領域・レーンを再確認してください。

動画は再生・一時停止・シーク・前後1フレーム・0.25〜2倍速に対応します。現在位置を解析開始時刻にできます。UI編集中も再生を継続します。UVCはライブプレビューとMP4録画に対応し、切断時は3秒間隔で再接続を試みます。音声は扱いません。

**Phase 1では分析開始ボタンは無効です。** ゲート・コブの確認（Phase 2）が必要であり、不足理由を画面下部に表示します。ミドルパートと全レーンを確定すれば、Phase 1のエリア設定は完了です。

## 保存

既定は作業フォルダ内の`data/calibration.sqlite`です。「設定を復元…」で別のSQLiteも開けます。
正規化座標・共有境界・確定状態・レーン属性・表示設定・ソース情報・解析開始位置・検出状況を保存します。保存ごとのスナップショットと編集履歴を保持します。Undoは直近100操作で、再起動後は引き継ぎません。
旧版の境界比率形式は読み込み時に正規化XY座標へ変換します。旧スナップショットは変更しません。修正版で保存した新形式は修正版アプリで復元してください。
元動画はコピーせずパスを参照します。移動した動画は元のパスに戻すか、新しい映像ソースとして設定してください。

録画はMP4（mp4v）と同名の`.timestamps.json`を生成します。JSONは各録画フレームのライブセッション内時刻（ミリ秒）です。欠落時は固定FPSのMP4時刻と一致しないため、この対応表を使います。

## カメラなしの動作確認

```powershell
.\.venv\Scripts\python.exe -m tools.create_test_video
.\.venv\Scripts\python.exe -m mogul --video outputs/synthetic-course.mp4
```

ラベル付き合成映像です。描画した点・旗・コブはAI認識結果ではありません。

## テストと診断

```powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pytest -v
.\.venv\Scripts\python.exe -m mogul --video outputs/synthetic-course.mp4 --smoke-seconds 3 --screenshot outputs/phase1.png
```

実機確認は通常のWindowsデスクトップで実行します。番号を省略すると一覧表示のみです。

```powershell
.\.venv\Scripts\python.exe -m tools.check_uvc
.\.venv\Scripts\python.exe -m tools.check_uvc --device 0 --width 640 --height 480 --fps 30 --seconds 2
```

既定の自動テストはカメラにアクセスしません。接続失敗時はWindowsのカメラプライバシー設定・他アプリの占有・対応モードを確認してください。隔離環境では列挙できてもDirectShow接続に失敗する場合があります。

## 構成

| モジュール | 役割 |
|---|---|
| `mogul/app.py`, `source_selector.py` | 日本語GUI、入力選択、操作の接続 |
| `mogul/video.py` | VideoSource、File/UVC、録画、別スレッド処理 |
| `mogul/editor.py` | MiddlePartEditor / LaneEditor、透過表示、マウス操作 |
| `mogul/models.py`, `geometry.py` | 正規化モデル、共有境界、形状・領域計算 |
| `mogul/workflow.py` | CalibrationWorkflow、Undo、AnalysisControllerの開始条件 |
| `mogul/detectors.py` | MiddlePartDetector / LaneDetectorの差し替え契約 |
| `mogul/repository.py` | EventRepository、SQLite保存・復元・履歴 |
| `tests/`, `tools/` | 回帰テスト、診断、合成動画生成、旧環境検証GUI |

既存環境のバージョンは[環境報告](ENVIRONMENT_V18.md)、今回の結果・制約・残作業は[Phase 1報告](PHASE1_REPORT.md)に記録しています。
`requirements.in`は直接依存、`requirements.txt`は固定バージョンです。今回の実装では依存追加なし。SQLiteはPython標準です。

AIモデル・学習データは同梱していません。採用時は前後処理、精度、CPU性能、メモリ、再配布条件を検証してください。
既存PATHのFFmpeg/ffprobeを使用します。FFmpeg 4.1.3は古く、すべてのコーデック・UVCを保証しません。実撮影の可変FPS・回転情報・HEVC/10bit・破損動画・長時間運用は追加検証が必要です。
`.venv/`、`data/`、`outputs/`、キャッシュはGitから除外します。
