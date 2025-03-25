## wndowsの場合
- `git clone https://github.com/pyenv-win/pyenv-win.git "%USERPROFILE%\.pyenv"`
- `pyenv install --list` で任意のバージョンの存在を確認
- reuquirement.txtに必要ライブラリを記載
- `init [pythonバージョン名（3.6.0など）]`　により環境作成、ライブラリインストール
- `venv\Scripts\activate`　で環境に入る
- `deactivate`　で環境を出る

## macの場合
- `git clone https://github.com/pyenv/pyenv.git ~/.pyenv`
- `pyenv install --list` で任意のバージョンの存在を確認
- reuquirement.txtに必要ライブラリを記載
- `./init.sh [pythonバージョン名（3.6.0など）]` により環境作成、ライブラリインストール
- `. venv/bin/activate` で環境に入る
- `deactivate` で環境を出る