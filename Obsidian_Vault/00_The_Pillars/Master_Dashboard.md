---
type: Pillar
status: Active
related_pillar: "[[Thesis_Master_Plan]]"
tags: [thesis, master_index, dashboard]
---
# ðŸ“Š Thesis Master Dashboard

This dashboard uses **Dataview** (if installed) to provide a live view of your Research DAG.

## ðŸŸ¢ Active Research Loops
*The questions currently being investigated.*

```dataview
TABLE status as "Status", related_pillar as "Chapter"
FROM "10_Inquiry"
WHERE status = "Active"
SORT created DESC
```

## ðŸ—ï¸ Methodologies & Logic
*Mathematical foundations and implementation details.*

```dataview
LIST
FROM "20_Logic_and_Method"
WHERE type = "Logic"
SORT file.name ASC
```

## ðŸ§ª Ongoing Trials
*Experimental runs in progress.*

```dataview
TABLE parameters as "Parameters", status as "State"
FROM "30_Trials"
WHERE status = "Active"
```

## ðŸ“œ Evidence & Findings
*Validated results ready for thesis redaction.*

```dataview
TABLE status as "Outcome", related_pillar as "Chapter"
FROM "40_Evidence"
WHERE type = "Evidence"
SORT file.mtime DESC
```

## ðŸ“š Critical Literature
*Key sources linked to current work.*

```dataview
LIST
FROM "90_Library"
WHERE tags.contains("literature")
LIMIT 10
```

---
**Note:** If you don't see tables above, ensure the **Dataview** plugin is enabled in your Obsidian settings.

