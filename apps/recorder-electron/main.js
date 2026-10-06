const { app, BrowserWindow, shell, Menu } = require('electron');
const { spawn } = require('child_process');
const path = require('path');

function parseArg(key, fallback = null) {
  const prefix = `--${key}=`;
  const hit = process.argv.find((arg) => arg.startsWith(prefix));
  if (!hit) return fallback;
  return hit.slice(prefix.length);
}

function hasFlag(flag) {
  return process.argv.includes(`--${flag}`);
}

function repoRootFromCwd() {
  // dev: run from `apps/recorder-electron`
  return path.resolve(process.cwd(), '..', '..');
}

function startBackend({ device, mock }) {
  const cwd = repoRootFromCwd();
  const env = {
    ...process.env,
    HAPTICA_HOME: app.getPath('userData'),
    HAPTICA_UI_DIST_DIR: path.join(cwd, 'apps', 'recorder-ui', 'dist'),
    HAPTICA_CONFIG_DIR: path.join(cwd, 'configs'),
  };
  const args = ['run', 'transcriptions', 'ui', `--device`, device, `--port`, '0', '--open', 'none'];
  if (mock) args.push('--mock');
  const proc = spawn('uv', args, { cwd, env });
  proc.stderr.on('data', (buf) => {
    // keep stderr for debugging
    process.stderr.write(buf);
  });
  return { proc };
}

function waitForUiUrl(proc, timeoutMs = 15000) {
  return new Promise((resolve, reject) => {
    const startedAt = Date.now();
    let settled = false;
    function maybeTimeout() {
      if (settled) return;
      if (Date.now() - startedAt > timeoutMs) {
        settled = true;
        reject(new Error('Timed out waiting for backend to print HAPTICA_UI_URL'));
      }
    }
    const interval = setInterval(maybeTimeout, 250);
    proc.stdout.setEncoding('utf8');
    proc.stdout.on('data', (chunk) => {
      if (settled) return;
      const lines = chunk.split(/\r?\n/);
      for (const line of lines) {
        if (line.startsWith('HAPTICA_UI_URL=')) {
          settled = true;
          clearInterval(interval);
          resolve(line.slice('HAPTICA_UI_URL='.length).trim());
          return;
        }
      }
    });
    proc.on('exit', (code) => {
      if (settled) return;
      settled = true;
      clearInterval(interval);
      reject(new Error(`Backend exited early (code ${code})`));
    });
  });
}

function createMenu({ getUrl }) {
  const template = [
    {
      label: 'Haptica Recorder',
      submenu: [
        { role: 'about' },
        { type: 'separator' },
        {
          label: 'Open In Chrome',
          click: async () => {
            const url = getUrl();
            if (!url) return;
            if (process.platform === 'darwin') {
              try {
                spawn('open', ['-a', 'Google Chrome', url], { stdio: 'ignore', detached: true });
                return;
              } catch (err) {
                // fall through
              }
            }
            await shell.openExternal(url);
          },
        },
        {
          label: 'Open In Browser',
          click: async () => {
            const url = getUrl();
            if (url) await shell.openExternal(url);
          },
        },
        { type: 'separator' },
        { role: 'quit' },
      ],
    },
    {
      label: 'View',
      submenu: [{ role: 'reload' }, { role: 'toggleDevTools' }],
    },
  ];
  Menu.setApplicationMenu(Menu.buildFromTemplate(template));
}

async function main() {
  const device = parseArg('device', 'dummy');
  const mock = hasFlag('mock') || device === 'dummy';

  let currentUrl = null;
  let backend = null;

  app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') app.quit();
  });

  app.on('before-quit', () => {
    if (backend?.proc && !backend.proc.killed) {
      backend.proc.kill('SIGTERM');
    }
  });

  await app.whenReady();
  createMenu({ getUrl: () => currentUrl });

  backend = startBackend({ device, mock });
  currentUrl = await waitForUiUrl(backend.proc);

  const win = new BrowserWindow({
    width: 1280,
    height: 800,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  await win.loadURL(currentUrl);
}

main().catch((err) => {
  console.error(err);
  app.quit();
});
