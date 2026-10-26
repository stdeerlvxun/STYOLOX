import { execFileSync } from 'node:child_process';
import process from 'node:process';

function runGit(args) {
  try {
    return execFileSync('git', args, {
      encoding: 'utf8',
      env: { ...process.env, GIT_TERMINAL_PROMPT: '0' },
      stdio: ['ignore', 'pipe', 'pipe'],
      timeout: 30_000
    }).trim();
  } catch (error) {
    const detail = String(error.stderr || error.message || '').trim();
    throw new Error(`git ${args.join(' ')} 执行失败${detail ? `：${detail}` : ''}`);
  }
}

function fail(message) {
  console.error(`DEPLOY_GATE_BLOCKED: ${message}`);
  process.exit(1);
}

try {
  const root = runGit(['rev-parse', '--show-toplevel']);
  process.chdir(root);

  const status = runGit(['status', '--porcelain=v1', '--untracked-files=all']);
  if (status) {
    fail('工作区存在未提交改动。请完成检查、提交并推送后再部署。');
  }

  const branch = runGit(['symbolic-ref', '--quiet', '--short', 'HEAD']);
  const remote = runGit(['config', `branch.${branch}.remote`]);
  const mergeRef = runGit(['config', `branch.${branch}.merge`]);
  if (!remote || !mergeRef.startsWith('refs/heads/')) {
    fail(`分支 ${branch} 没有有效的远端跟踪配置。`);
  }

  const remoteUrl = runGit(['remote', 'get-url', remote]);
  const githubRemote = /^(?:git@github\.com:|ssh:\/\/git@github\.com\/|https:\/\/github\.com\/)/i;
  if (!githubRemote.test(remoteUrl)) {
    fail(`远端 ${remote} 不是已确认的 GitHub 仓库。`);
  }

  runGit(['fetch', '--quiet', remote, mergeRef]);

  const localHead = runGit(['rev-parse', 'HEAD']);
  const remoteLine = runGit(['ls-remote', '--exit-code', remote, mergeRef]);
  const remoteHead = remoteLine.split(/\s+/)[0];
  if (!remoteHead || localHead !== remoteHead) {
    fail(`本地提交 ${localHead.slice(0, 12)} 尚未与 GitHub 远端分支保持一致。`);
  }

  const repoMatch = remoteUrl.match(/github\.com[/:]([^/]+\/[^/]+?)(?:\.git)?$/i);
  const repository = repoMatch ? repoMatch[1] : 'github.com';
  console.log('DEPLOY_GATE_OK');
  console.log(`REPOSITORY=${repository}`);
  console.log(`BRANCH=${branch}`);
  console.log(`COMMIT=${localHead}`);
} catch (error) {
  fail(error.message);
}
