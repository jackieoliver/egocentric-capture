const { contextBridge } = require('electron');

contextBridge.exposeInMainWorld('haptica', {
  version: '0.1.0',
});

