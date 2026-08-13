import { app, BrowserWindow, Menu, session, shell } from "electron";

const applicationUrl = new URL(
  process.env.STRUCTUREFIRST_DESKTOP_URL ?? "http://127.0.0.1:5173",
);

// Electron documents this switch for selecting the discrete adapter on
// dual-GPU systems. It must be set before the ready event.
app.commandLine.appendSwitch("force_high_performance_gpu");

async function createWindow(): Promise<void> {
  const window = new BrowserWindow({
    width: 1540,
    height: 980,
    minWidth: 1180,
    minHeight: 720,
    backgroundColor: "#0d1419",
    autoHideMenuBar: true,
    show: false,
    title: "StructureFirst",
    webPreferences: {
      contextIsolation: true,
      sandbox: true,
      nodeIntegration: false,
      webSecurity: true,
      allowRunningInsecureContent: false,
      webviewTag: false,
    },
  });

  window.webContents.setWindowOpenHandler(({ url }) => {
    if (isSafeExternalUrl(url)) void shell.openExternal(url);
    return { action: "deny" };
  });
  window.webContents.on("will-navigate", (event, url) => {
    if (isApplicationUrl(url)) return;
    event.preventDefault();
    if (isSafeExternalUrl(url)) void shell.openExternal(url);
  });
  window.once("ready-to-show", () => window.show());

  await waitForApplication(applicationUrl);
  await window.loadURL(applicationUrl.toString());
}

function isApplicationUrl(value: string): boolean {
  try {
    return new URL(value).origin === applicationUrl.origin;
  } catch {
    return false;
  }
}

function isSafeExternalUrl(value: string): boolean {
  try {
    return new URL(value).protocol === "https:";
  } catch {
    return false;
  }
}

async function waitForApplication(url: URL): Promise<void> {
  for (let attempt = 0; attempt < 120; attempt += 1) {
    try {
      const response = await fetch(url, {
        signal: AbortSignal.timeout(1_000),
      });
      if (response.ok) return;
    } catch {
      // The local TypeScript server and renderer can still be starting.
    }
    await new Promise((resolveDelay) => setTimeout(resolveDelay, 500));
  }
  throw new Error(`StructureFirst did not start at ${url.origin}.`);
}

app.whenReady().then(async () => {
  Menu.setApplicationMenu(null);
  session.defaultSession.setPermissionCheckHandler(() => false);
  session.defaultSession.setPermissionRequestHandler((_webContents, _permission, callback) => {
    callback(false);
  });
  await createWindow();
  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) void createWindow();
  });
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});
