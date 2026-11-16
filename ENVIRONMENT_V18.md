# v1.8 開発環境差分検証（2026-11-16）

## 判定

Phase 1のGUI・座標計算・動画処理の実装を開始できる状態です。
AI候補検出はOpenCV DNN / CPUを初期バックエンドとする方針です。ただし、対象領域・レーンの学習済みモデル、教師データ、精度基準は未指定であり、実モデル推論の動作・速度・必要メモリは未確認です。AI認識を含む全機能の準備完了を意味しません。
アプリ本体は実装していません。tools/以下は使い捨て可能な環境検証コードです。

## 維持した環境

既存.venvを削除・再作成せず使用。既存ライブラリの更新なし。
Gitには既存の未追跡ファイルがあり、そのまま維持しました。コミット・Git設定変更なし。

| 項目 | 実測 |
|---|---|
| Windows | NT 10.0.26300 / 26H2 / build 26300.9974 |
| OS表記 | 既存記録はWindows 11。レジストリProductNameはWindows 10 Home。CIMは権限エラーのためSKU照合未確認 |
| Python | 3.13.12 / Windows x64 / 既存.venv |
| pip | 26.3 |
| PySide6 / Qt | 6.12.0 / 6.12.0 |
| OpenCV | opencv-python 5.0.0.93 |
| NumPy | 2.5.4 |
| Pandas | 3.1.1 |
| openpyxl | 3.1.5 |
| pytest | 9.2.0 |
| Git | 2.54.0.windows.1 |
| FFmpeg / ffprobe | 既存PATH版4.1.3 |
| OpenCV内部FFmpeg | avcodec 61.19.100 / avformat 61.7.100 |

既存の推移的依存も含む一覧はrequirements.txtに記録。PySide6_Addons、Essentials、shiboken6は6.12.0、Pdf/WebEngineは6.12.0.140。その他の既存パッケージはcolorama 0.4.6、et_xmlfile 2.0.0、iniconfig 2.3.1、packaging 26.3、pluggy 1.7.0、Pygments 2.21.0、python-dateutil 4.0.0、tzdata 2026.6です。

## 追加

- pytest-qt 4.5.0：Qtイベントとマウス操作の検証に必要。
- typing_extensions 4.16.0：pytest-qtの追加依存。

requirements.inへ直接依存を追記、requirements.txtへ検証済み環境を固定。
Shapely、PyAV、PyTorch、ONNX Runtime、Ultralyticsは追加していません。
凸四角形に範囲を限定した環境検証ではNumPy/OpenCVで十分でした。任意の凹形状・複数領域のブーリアン演算・頑健なトポロジー処理が必要になればShapelyを再検討します。

## 実行済み検証

`.venv/Scripts/python.exe -m pytest -v`：8件成功。
`.venv/Scripts/python.exe -m pip check`：競合なし。
`.venv/Scripts/python.exe -m tools.v18_probe --seconds 3`：終了コード0。
Qtプラットフォームはwindowsであり、offscreenではありません。

- 既存4件：Qtイベントループ、画像往復、Excel往復、FFmpeg H.264動画→OpenCV。
- 追加GUI：生成画像表示、半透明ポリゴン、四隅、Qtマウスイベントによる実ドラッグ、形状更新、シングルクリック確定、ダブルクリック再編集、2/3/4レーン表示切替、JSON保存復元。
- API：QGraphicsView/Scene/PolygonItem/EllipseItem/Item/SceneMouseEvent、QPainter、QColor、QTimer、Signal/Slotをインポートし、検証ウィンドウで使用可能。SceneMouseEventはインポート確認。
- 座標：頂点管理、正規化と解像度変更、交差した四角形の拒否、点包含と境界、ポリゴン包含の面積、重複、共有辺の面積0、透視変換。凸四角形の前提で検証。
- 動画：合成H.264 / yuv420p / 160x120 / 10fps / 2秒をMP4とMOVで検証。ffprobeの時間とOpenCVのFPS・寸法・20フレーム、1秒へのシーク、フレーム取得、メディア時刻との整合を確認。
- 同期：CFR合成動画の取得タイムスタンプからマーカーを選ぶ計算を検証。動画再生中の画面オーバーレイ同期は未実装・未検証。
- 表示のキャプチャ：outputs/v18-probe.png（生成物、Git対象外）。人による目視・手動操作は未実施。

手動確認：`.venv/Scripts/python.exe -m tools.v18_probe`。
四隅をドラッグし、クリックで固定、ダブルクリックで解除できます。レーン数・保存復元はテストから操作します。

## AI / GPU

- NVIDIA GeForce RTX 3050 Laptop GPU、4096 MiB、ドライバ551.23をnvidia-smiで確認。
- nvidia-smiのCUDA表示は12.4。これはドライバの対応表示であり、CUDA Toolkitや推論ランタイムの導入を示しません。nvccはPATH上にありません。
- OpenCVのCUDAデバイス数は0。現状でCUDA推論が使えるとは判定できません。
- torch / onnxruntime / ultralyticsは未導入。
- OpenCV DNNのONNX読込APIあり。OpenCVバックエンドの利用可能ターゲットにCPU（0）を確認。
- Phase 1はCPUを基準に、適合するONNXモデルを選定後に読込・推論・精度・速度を測定します。一般的な人物検出モデルがコース領域・レーンを認識できる保証はありません。領域候補は輪郭・線分等の古典処理も比較対象です。
- YOLO系はモデルとエクスポート演算の互換性、重みの再配布条件を確認してから採用。Ultralyticsの公式条件はAGPL-3.0またはEnterprise。今回は重みの取得・配布なし。
- 推論メモリはモデル未指定のため実測不可。入力640x640 RGB float32だけで約4.69 MiB、1080p BGR uint8は約5.93 MiB/フレーム。これに重み・中間テンソル・作業領域・GUI/動画バッファが加わります。VRAM 4GBだけでは適合性は判断できません。バッチ1でCPU RSS/ピーク・GPU使用量・遅延をモデル採用時に測定してください。

参照（2026-11-16確認）：
- https://docs.opencv.org/doc/doxygen/html/d6/d0f/group__dnn.html
- https://docs.opencv.org/doc/doxygen/html/de/db7/api_dnn_engine_selection.html
- https://www.ultralytics.com/license

## 未確認・次の実装時の対象

- UVC機器の列挙・接続・ライブ取得・録画ファイルへの切替：DirectShowビルド有効は確認。実機アクセスは未実施。
- 実際の撮影動画、可変フレームレート、回転情報、HEVC/10bit、破損ファイル、長時間再生、音声同期。MP4/MOVはコンテナであり、すべてのコーデックの対応を保証しません。
- ドラッグ矩形による範囲選択、レーンの個別手動補正、動画に重ねたマーカー同期：本体実装の対象で、今回の検証GUIには未実装。
- 任意ポリゴンの頑健な自己交差・包含・共通境界処理と保存形式の正式仕様。
- 学習済みモデル推論、AI候補精度、人物追跡・ゲート検出・コブ解析。
- 古いPATH版FFmpegは維持。配布時は実機コーデックとライセンス、配置方法を別途確認。
