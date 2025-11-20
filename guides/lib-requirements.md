# コードエージェント向けインストラクション（Python特化：ライブラリ適格性の自動検証と改善提案）

目的：現在のPythonプロジェクトが「一般的なライブラリ」の要件を満たしているか自動で検証し、満たしていない点について具体的に「何を追記・修正すればライブラリになるか」を差分（patch）で提示するための手順・チェックリスト・出力仕様。

---

1) 前提入力（エージェントが必ず受け取るもの）
- リポジトリのルート（ローカルパスまたはGit URL）
- 主言語：python（固定）
- 主要ブランチ名（デフォルト：main）
- （任意）期待するパッケージ名／公開先（例：PyPI）

---

2) 出力形式（必須）
エージェントは以下JSONオブジェクトを生成すること（キー順不問）：

{
  "summary": {"score": 0.0, "status": "pass|partial|fail", "notes": "..."},
  "checks": [
    {"id":"clear_responsibility", "result":"pass|partial|fail", "details":"...", "recommendations":[ "..."]},
    ...
  ],
  "suggested_changes": [
    {"file":"README.md", "patch":"---\n+++...", "reason":"..."},
    ...
  ],
  "templates": {"README.md":"...", "pyproject.toml":"...", "setup.cfg":"...", "tox.ini":"..."},
  "confidence": 0.0
}

要点：scoreは0〜100の合成スコア。suggested_changesには実際に適用可能な差分（unified diff形式）を含めること。templatesには言語特有の最小雛形を文字列で提供すること。

---

3) 合格基準（主要チェック項目：Python向け）
各項目は pass|partial|fail を返し、partial/fail には必須の具体的修正案を含める。

1. 明確な責務（clear_responsibility）
   - ソースを解析してライブラリの主機能を自然文で要約（1-2行）。
   - 条件：機能が1〜3個の明確なドメインで記述されている。

2. 公開APIの整備（public_api）
   - パッケージのトップレベル（package/__init__.py）を確認し、__all__や明示的な公開シンボルがあるか検出。
   - 条件：公開シンボルが整理されていること。なければ推奨 __all__ の差分を提示。

3. 依存関係の妥当性（dependencies）
   - pyproject.toml、requirements.txt、setup.cfg を読み取り依存一覧を作成。
   - 条件：プロジェクト固有の重い依存（Djangoフルスタック、巨大なC拡張等）が用途に比して正当化されていない場合は警告。

4. 独立でビルド・テスト可能（buildable_testable）
   - ビルドコマンド（python -m build）やテストコマンド（pytest）が存在し、ローカルで実行可能かを試行。実行不可の場合は静的チェックを行い実行手順を提示。

5. パッケージ化・配布可能性（packagable）
   - pyproject.toml または setup.py / setup.cfg の有無と必須メタ情報（name, version, license, description, authors）を確認。不足があればテンプレートを提供。

6. ドキュメント（docs）
   - README.md の存在と最低限のセクション（概要、インストール、使用例、API、ライセンス）を確認。欠落部分は差分で追加。

7. テストと品質（tests_quality）
   - tests/ または test_*.py の存在、pytest の構成、簡易カバレッジ推定。テストが存在しない場合は最小テストの雛形を生成。

8. バージョニングとリリース方針（versioning）
   - バージョンが pyproject.toml、__version__、もしくは Git タグで管理されているか確認。SemVer 推奨。

9. ライセンス（license）
   - LICENSE ファイルの有無を確認。無ければリスク提示と推奨ライセンス雛形を生成。

10. エラーハンドリングと安定したAPI仕様（error_handling）
    - 例外設計が一貫しているか（カスタム例外の定義やドキュメント化）。不在なら推奨設計を提案。

11. 互換性・拡張性（backwards_compatibility）
    - 公開APIを明確にし、deprecated 方針の記述があるか確認。なければ提言。

12. セキュリティ（security）
    - 依存の脆弱性スキャン（safety DB 等）に基づく警告、及び機密情報（.env、シークレット）や鍵が含まれていないかの検査。

---

4) 実行手順（ワークフロー：Python専用）
1. clone → detect python（pyproject.toml / setup.py / requirements.txt を確認）
2. 静的解析：ast解析でトップレベル関数/クラスを抽出、__all__ の存在確認
3. 依存解析：pyproject.toml / requirements.txt を読み取り依存リスト作成
4. ドキュメント確認：README.md, LICENSE の有無と主要セクション検出
5. テスト実行：pytest 実行を試みる（仮想環境で依存をインストールして実行）。実行不可時は失敗ログと静的推定
6. ビルド試行：python -m build を試行し dist/*.whl または sdist が生成されるか確認
7. 脆弱性スキャン：safety または pip-audit 等の出力を統合
8. 公開API評価：package/__init__.py の公開シンボル、一貫性、private 命名（_で始まる）違反を検出
9. 差分生成：不足点ごとに最小差分（unified diff）を作成
10. JSONレポート出力（前述フォーマット）

---

5) 具体的検査ロジックとコマンド例（実用指示）
- 静的解析：Python ast モジュールでトップレベル定義を列挙。__all__ を優先して公開APIを抽出。なければ public 名前規約（no leading underscore）から推定。
- 依存スキャン：pyproject.toml または requirements.txt を解析。pip-audit / safety のコマンドを実行して既知脆弱性を検出。
- テスト実行例：仮想環境を作成し、pip install -e .[test] または pip install -r requirements-dev.txt を試み、pytest -q を実行。
- ビルド試行例：pip install build; python -m build。生成される wheel/sdist の存在を確認。
- Lint/静的解析例：pylint, ruff, black の設定有無を確認し、推奨設定ファイルをテンプレートとして出す。

---

6) 推奨差分（例）
suggested_changes に含める差分は unified diff 形式で最小限にする。例として追加する代表的差分（説明のみ、差分自体はJSON内のpatchフィールドに入れる）：
- README.md の追加（概要、インストール、最小使用例、APIサマリ、ライセンス）
- pyproject.toml の最小雛形追加（name, version, description, authors, license, classifiers, dependencies）
- package/__init__.py に __all__ と __version__ を追加
- tests/test_basic.py の最小テスト追加（import と主要関数の簡単なアサーション）
- .github/workflows/ci.yml の最小CI追加（pytest と build を実行）

---

7) テンプレート（最小雛形の文字列提供）
templates に含める代表例（内容は文字列。実際のテンプレートはプロジェクトに合わせて微調整）：
- README.md：プロジェクト名、概要、インストール、使い方、ライセンス
- pyproject.toml：[project] セクション（name, version, description, authors, license）、build-system（setuptools / hatch / poetry 指定）
- setup.cfg / setup.py の場合は minimal の提示
- tests/test_basic.py の最小テスト雛形
- .github/workflows/ci.yml：Ubuntu 上で Python セットアップ、依存インストール、pytest、python -m build を実行するワークフロー

---

8) CI / 自動検証の雛形（GitHub Actions：文字列として提供）
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install build pytest
          if [ -f requirements-dev.txt ]; then pip install -r requirements-dev.txt; fi
      - name: Run tests
        run: pytest -q
  pack:
    runs-on: ubuntu-latest
    needs: test
    steps:
      - uses: actions/checkout@v4
      - name: Build package
        run: |
          pip install build
          python -m build

（注：上は単一ワークフローのサンプル。実際は環境や extras 要件に合わせて修正する）

---

9) スコアリング例（Python専用の重み）
- 明確な責務：20
- 公開API整備：20
- ドキュメント：15
- パッケージ化可能：15
- テストと品質：10
- 依存とセキュリティ：10
- ライセンス：10
合計100。各チェックで pass=full weight, partial=half weight, fail=0。

---

10) レポートの出力要件
- 最重要箇所トップ3（優先度順）を summary.notes に明記すること。
- checks 配列は各チェックごとに details（検査ログの抜粋）と recommendations（1つ以上の具体案）を含むこと。
- suggested_changes は atomic な patch 単位に分けること（1 patch = 1 concern）。
- templates は即コピペ可能な雛形文字列を含めること。

---

11) 自動修正方針（LLM を用いる場合の制約）
- 自動で API を破壊する変更は禁止。互換性を壊す修正は patch として出す場合でも "partial" とし、代替の非破壊案を必ず併記すること。
- テストが存在しない場合は最小テストを追加してから他の修正を適用する順序を推奨する。
- 重大なセキュリティリスク（平文シークレット等）が見つかった場合は即時 fail とし、修正手順を明確に記述する。

---

12) 実装チェックリスト（エージェントが即実行する短い手順）
1. git clone リポジトリ
2. 言語確認（pyproject.toml 等）
3. ast 解析でトップレベル定義と __all__ を抽出
4. 依存関係リスト作成（pyproject.toml / requirements.txt）
5. README, LICENSE の有無確認
6. pytest 実行（仮想環境内）→ 成功/失敗ログ収集
7. python -m build を試行 → アーティファクト確認
8. pip-audit / safety 実行（脆弱性検出）
9. 差分（unified diff）を生成し suggested_changes に追加
10. JSON レポートを出力

---

13) 追加の注意点
- 差分は常に最小変更を心掛ける（可逆的な変更が望ましい）。
- 提案は実行可能なコマンドと具体的ファイルパスを含めること。
- 実行時に外部ネットワークアクセスが不可の場合は静的チェックにフォールバックし、その旨を confidence に反映すること。

---

必要であれば、このPython特化のインストラクションに基づいて「lib-validator」Python スクリプト群と GitHub Action を出力します。どの出力形式（単体スクリプト / package / GitHub Action）を優先しますか？
