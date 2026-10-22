; Inno Setup 6 script for the Windows installer. Saved as UTF-8 with a BOM,
; which is how Inno Setup knows the translated messages below are UTF-8.
; Built by packaging/build.py, which defines everything below from the command
; line:
;
;   ISCC /DAppVersion=0.1.0 /DAppExeName=YOLOAssistant.exe
;        /DSourceDir=<PyInstaller folder> /DRepoRoot=<checkout>
;        /DIconFile=<.ico> /DOutputDir=<dist> installer.iss
;
; The install is per user and needs no administrator rights, like everything
; else the app installs: it goes under %LOCALAPPDATA%\Programs, and the managed
; Python environment goes under %LOCALAPPDATA%\YOLOAssistant.

#ifndef AppVersion
  #error Build this with packaging/build.py, which defines AppVersion and the paths.
#endif

#define AppName "YOLO Assistant"
; Everything app/core/paths.py writes, on Windows.
#define DataDir "{localappdata}\YOLOAssistant"

[Setup]
; Never change this: it is how an upgrade finds the installation it replaces.
AppId={{5C2B8E4A-7F1D-4C36-9B0E-3A6D2F8E1C47}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
LicenseFile={#RepoRoot}\LICENSE
SetupIconFile={#IconFile}
UninstallDisplayIcon={app}\{#AppExeName}
UninstallDisplayName={#AppName}
OutputDir={#OutputDir}
OutputBaseFilename=YOLOAssistant-{#AppVersion}-setup
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
; An upgrade over a running copy would replace files it has open.
CloseApplications=yes
RestartApplications=no
; Recent Inno Setup 6 releases turn on Windows' RedirectionGuard for Setup by
; default (see the RedirectionGuard directive in its help), and
; the app started from the last page ("Launch ...") inherits it, along with
; every process the app starts. RedirectionGuard refuses to follow a junction a
; non-administrator created -- and uv reaches its managed Python through one
; (pythons\cpython-3.12-windows-x86_64-none). So that copy of the app cannot
; start a single worker: "uv trampoline failed to spawn Python child process",
; while the same app started from the Start menu works. The mitigation guards
; an *elevated* installer against junctions planted by an unprivileged user;
; this one runs as that user (PrivilegesRequired=lowest) and has nothing to
; escalate to, so it costs nothing to turn off.
RedirectionGuard=no

[Languages]
Name: "en"; MessagesFile: "compiler:Default.isl"
Name: "de"; MessagesFile: "compiler:Languages\German.isl"
Name: "pl"; MessagesFile: "compiler:Languages\Polish.isl"
Name: "ru"; MessagesFile: "compiler:Languages\Russian.isl"

[CustomMessages]
en.RemoveEnvironment=Also remove the Python environment, downloaded models and logs the app installed in%n%n%1?%n%nThey are often more than 10 GB. Your projects in its "projects" folder are kept either way.%n%nChoose No to keep them for a later reinstall.
de.RemoveEnvironment=Auch die Python-Umgebung, die heruntergeladenen Modelle und die Protokolle entfernen, die die App hier installiert hat?%n%n%1%n%nSie belegen oft mehr als 10 GB. Ihre Projekte im Ordner "projects" bleiben in jedem Fall erhalten.%n%nWählen Sie Nein, um sie für eine spätere Neuinstallation zu behalten.
pl.RemoveEnvironment=Czy usunąć również środowisko Pythona, pobrane modele i dzienniki, które aplikacja zainstalowała w%n%n%1?%n%nCzęsto zajmują ponad 10 GB. Projekty w folderze "projects" zostaną zachowane w każdym przypadku.%n%nWybierz Nie, aby zachować je na potrzeby ponownej instalacji.
ru.RemoveEnvironment=Удалить также окружение Python, загруженные модели и журналы, которые приложение установило в%n%n%1?%n%nОбычно они занимают больше 10 ГБ. Ваши проекты в папке "projects" сохранятся в любом случае.%n%nВыберите «Нет», чтобы оставить их для повторной установки.

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[InstallDelete]
; A file a previous version shipped and this one does not would otherwise
; linger in _internal and could be loaded in preference to its replacement.
Type: filesandordirs; Name: "{app}\_internal"

[Icons]
Name: "{autoprograms}\{#AppName}"; Filename: "{app}\{#AppExeName}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExeName}"; Description: "{cm:LaunchProgram,{#AppName}}"; Flags: nowait postinstall skipifsilent

[Code]
// What the app keeps under its data directory, minus the user's own work.
// "projects" is deliberately absent: it is the default home for datasets and
// trained models, and an uninstaller has no business deleting those.
const
  OwnedFolders = 'tools,pythons,venvs,cache,workers,logs';
  OwnedFiles = 'state.json,settings.json';

procedure RemoveOwned(Root: String);
var
  Names: TStringList;
  I: Integer;
begin
  Names := TStringList.Create;
  try
    Names.CommaText := OwnedFolders;
    for I := 0 to Names.Count - 1 do
      DelTree(Root + '\' + Names[I], True, True, True);
    Names.CommaText := OwnedFiles;
    for I := 0 to Names.Count - 1 do
      DeleteFile(Root + '\' + Names[I]);
  finally
    Names.Free;
  end;
  // Only succeeds if empty, which is the point: an empty projects folder
  // goes, one with work in it stays, and the root follows only if both went.
  RemoveDir(Root + '\projects');
  RemoveDir(Root);
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  Root: String;
begin
  if CurUninstallStep <> usPostUninstall then
    Exit;
  Root := ExpandConstant('{#DataDir}');
  if not DirExists(Root) then
    Exit;
  // A silent uninstall (an upgrade script, a deployment tool) keeps the data:
  // deleting gigabytes is not something to do without being asked.
  if UninstallSilent then
    Exit;
  if SuppressibleMsgBox(FmtMessage(CustomMessage('RemoveEnvironment'), [Root]),
                        mbConfirmation, MB_YESNO or MB_DEFBUTTON2, IDNO) = IDYES then
    RemoveOwned(Root);
end;
