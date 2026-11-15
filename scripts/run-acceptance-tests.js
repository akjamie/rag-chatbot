import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';

const venvPyWin = path.resolve('venv/Scripts/python.exe');
const venvPyNix = path.resolve('venv/bin/python');
let cmd = 'python -m pytest';
if (fs.existsSync(venvPyWin)) {
  cmd = `"${venvPyWin}" -m pytest`;
} else if (fs.existsSync(venvPyNix)) {
  cmd = `"${venvPyNix}" -m pytest`;
}

const args = process.argv.slice(2);
const testTargets = args.length > 0 ? args.join(' ') : 'tests/config/test_dependencies.py tests/handler/test_generic_query_handler.py tests/config/test_startup.py tests/handler/test_reranker.py';
const fullCmd = `${cmd} ${testTargets}`;

const res = spawnSync(fullCmd, { stdio: 'inherit', shell: true });
process.exit(res.status ?? 0);
