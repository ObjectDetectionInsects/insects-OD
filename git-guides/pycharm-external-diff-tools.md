# External Diff Tools

## Specify external tools for comparing or merging files and folders, and associated settings.

### Windows and Linux:
1.  Open PyCharm IDE
2.  Click "File" on toolbar
3.  Choose "Settings"
4.  Tools 
5.  Diff & Merge
6.  External Diff Tools

Add a new external tool. In the dialog the opens, configure the following options:

- **Tool group:** select whether you want to use a diff or merge tool.

- **Program path:** specify the path to the executable file of the tool you want to use.

    For example: **C:\Program Files\Beyond Compare 4\BCompare.exe on Windows or /Applications/Beyond Compare.app/Contents/MacOS/bcomp** on macOS.

- **Tool name:** enter the name of the external tool that you're configuring.

- **Argument pattern:** set the diff tool parameters.

    Specify the necessary parameters in the proper order:

        - %1: left (local changes)

        - %2: right (server changes)

        - %3: base (the current version without the local changes)

        - (merge tool only) %4: output (merge result)

- (merge tool only) **Trust process exit code:** select to silently finish the merge if the **exitCode** of the external merge tool is
   set to **0** (successful). Otherwise, you will be prompted to indicate the success of the resolution after the tool has exited.


   ![example](https://dev.azure.com/ORB-PCB/5f0c4c94-33c2-4109-be4b-9f456426080b/_apis/git/repositories/9c87839a-4e60-49f7-ba66-e1e007151dc1/items?path=/assets/pycharm-byd-compare.PNG&versionDescriptor%5BversionOptions%5D=0&versionDescriptor%5BversionType%5D=0&versionDescriptor%5Bversion%5D=master&resolveLfs=true&%24format=octetStream&api-version=5.0)
