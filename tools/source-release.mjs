// Source-only publication gate. Runtime models, snapshots and credentials are never packaged.
import {execFileSync} from 'node:child_process';
import * as fs from 'node:fs';
import path from 'node:path';
import {createHash, randomUUID} from 'node:crypto';
const git = (...args) => execFileSync('git', ['-c', 'core.quotepath=false', ...args], {encoding: 'utf8'}).trim();
const sha256 = data => createHash('sha256').update(data).digest('hex');
const mode = process.argv[2];
if (!['check', 'package'].includes(mode)) throw Error('Choose check or package');
if (git('status', '--porcelain')) throw Error('Working tree must be clean');
if (git('remote', 'get-url', 'origin') !== 'git@github.com:stdeerlvxun/STYOLOX.git') throw Error('Repository mismatch');
const commit = git('rev-parse', 'HEAD');
const branch = git('branch', '--show-current');
if (!branch) throw Error('Named branch required');
const remote = git('ls-remote', '--exit-code', 'origin', `refs/heads/${branch}`).split(/\s/)[0];
if (remote !== commit) throw Error('Local and GitHub commits differ');
const files = git('ls-tree', '-r', '--name-only', 'HEAD').split('\n');
const allowed = /\.(?:py|md|txt|html|css|js|cjs|mjs|cmd|ps1|sh|toml|ini|yml|yaml|service|timer|example)$/i;
for (const file of files) {
  if (!allowed.test(file) && !['.gitignore', 'package.json', 'source-provenance.json'].includes(file)
      && !file.endsWith('/.gitignore')) throw Error(`Non-source file: ${file}`);
  const text = execFileSync('git', ['show', `HEAD:${file}`], {encoding: 'utf8', maxBuffer: 8*1024*1024});
  if (text.includes('\0') || /-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|gh[pousr]_[A-Za-z0-9]{30,}|AKIA[0-9A-Z]{16}/.test(text)) {
    throw Error(`Sensitive or binary content: ${file}`);
  }
}
console.log(JSON.stringify({gate: 'passed', commit, branch, source_files: files.length}));
if (mode === 'package') {
  const directory = path.resolve(process.env.STYOLOX_RELEASE_DIR || '../STYOLOX-release-artifacts');
  if (directory === process.cwd() || directory.startsWith(process.cwd()+path.sep)) throw Error('Package outside source checkout');
  fs.mkdirSync(directory, {recursive: true});
  const id = new Date().toISOString().replace(/[:.]/g, '-')+'-'+randomUUID().slice(0, 8);
  const temp = path.join(directory, '.pending-'+id+'.zip');
  execFileSync('git', ['archive', '--format=zip', `--output=${temp}`, commit]);
  const data = fs.readFileSync(temp);
  if (data.length < 100 || data.readUInt32LE(0) !== 0x04034b50) throw Error('Invalid archive');
  const record = {repository: 'stdeerlvxun/STYOLOX', commit, generated_at: new Date().toISOString(),
    file: 'styolox-latest.zip', size: data.length, sha256: sha256(data), source_files: files.length, scope: 'public-source-only'};
  const pending = path.join(directory, '.pending-'+id+'.json');
  fs.writeFileSync(pending, JSON.stringify(record, null, 2)+'\n', {flag: 'wx'});
  const latest = path.join(directory, record.file), metadata = path.join(directory, 'release.json');
  const history = path.join(directory, 'history', id);
  fs.mkdirSync(history, {recursive: true});
  const moved = [];
  try {
    for (const [source, name] of [[latest, `styolox-${id}.zip`], [metadata, `release-${id}.json`]]) {
      if (fs.existsSync(source)) { const dest=path.join(history, name); fs.renameSync(source, dest); moved.push([source, dest]); }
    }
    fs.renameSync(temp, latest); fs.renameSync(pending, metadata);
  } catch (error) {
    // Restore the previous release if replacement could not finish.
    if (fs.existsSync(latest) && !fs.existsSync(temp)) fs.renameSync(latest, temp);
    for (const [source, dest] of moved.reverse()) fs.renameSync(dest, source);
    throw error;
  }
  console.log(JSON.stringify({package: latest, ...record}));
}
