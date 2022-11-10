
## Use `AB#` mention to link from Git to Azure Boards work items 

From a Git commit, pull request or issue, use the following syntax to create a link to your Azure Boards work item. Enter the `AB#ID` within the text of a commit message. Or, for a pull request or issue, enter the `AB#ID` within the title or description (not a comment).

From a Git commit or pull request, use the following syntax to create a link to your Azure Boards work item. Enter the `AB#ID` within the text of a commit message or for a pull request, enter the `AB#ID` within the pull request title or description (not a pull request comment). 

```
AB#{ID}
```

For example, `AB#125` will link to work item ID 125. 

You can also enter a commit or pull request message to transition the work item. The system will recognize `fix, fixes, fixed` and apply it to the #-mention item that follows. Some examples are provided as shown. 

Examples: 

| Commit message                              | Action |
| :------------------------------------------ | :----------------------------------------------- |
| `Fixed AB#123`                              | Links and transitions the work item to the "done" state. |
| `Adds a new feature, fixes AB#123.`         | Links and transitions the work item to the "done" state. |
| `Fixes AB#123, AB#124, and AB#126`          | Links to Azure Boards work items 123, 124, and 126. Transitions only the first item, 123 to the "done" state. |
| `Fixes AB#123, Fixes AB#124, Fixes AB#125` | Links to Azure Boards work items 123, 124, and 126. Transitions all items to the "done" state. |
| `Fixing multiple bugs: issue #123 and user story AB#234` | Links to Git issue 123 and Azure Boards work item 234. No transitions. |


