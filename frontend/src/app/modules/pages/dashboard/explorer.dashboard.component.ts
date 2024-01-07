import { AgGridAngular } from '@ag-grid-community/angular';
import { GridOptions, GridReadyEvent } from '@ag-grid-community/core';
import { Component, OnInit, ViewChild } from '@angular/core';
import { Observable } from 'rxjs';
import { Option } from 'src/models/option';
import { OptionsService } from 'src/services/options.service';
import { TitleCasePipe } from '@angular/common';
var Plotly = require("plotly.js-dist-min")

class OptionParameters {
    ticker: string = "TSLA"
    optionType: string = "Call"
    field: string = "price"
    range: number = 5
}

@Component({
    selector: 'explorer-dashboard',
    templateUrl: './explorer.dashboard.component.html',
    styleUrls: ["./explorer.dashboard.component.scss"],
    host: { class: 'myClass' },
})
export class ExplorerDashboardComponent implements OnInit {

    constructor(private optionsService: OptionsService, private titlecasePipe: TitleCasePipe) {

    }

    ngOnInit() {
        Plotly.newPlot('divSurfaceOptions', this.graph.data, this.graph.layout as Plotly.Layout, {
            scrollZoom: true
        });
    }

    public rowData$!: Observable<Option[]>;

    @ViewChild(AgGridAngular) agGrid!: AgGridAngular;


    public graph = {
        data: [{ x: [], y: [], z: [], type: 'surface', showscale: false } as Plotly.Data,
        ],
        layout: {
            autosize: true,
            margin: { l: 0, r: 0, b: 0, t: 0 },
            scene: {
                xaxis: { title: "Maturity", type: 'date' },
                yaxis: { title: "Strike" },
                zaxis: { title: "" },
            },
        },
    }
    spot = 0

    onGridReady(params: GridReadyEvent) {
        this.loadOptions();
    }

    fields = ["price", "delta", "gamma", "vega", "theta", "rho", "implied_vol"]
    ranges = [5, 10, 15, 25]
    defaultOptionType = "Call"

    parameters = new OptionParameters()

    public gridOptions: GridOptions = {

        defaultColDef: {
            sortable: true,
            filter: true,
            type: 'rightAligned',
            flex: 1, floatingFilter: true
        },
        columnDefs: [
            { field: 'type', type: 'leftAligned', hide: true },
            { field: 'strike' },
            { field: 'maturity', type: 'leftAligned', },
            { field: 'price' },
            { field: 'delta' },
            { field: 'gamma' },
            { field: 'vega' },
            { field: 'theta' },
            { field: 'rho' },
            { field: 'impliedVol' },
        ]
    }


    onFieldChange(event: any) {
        this.parameters.field = event.target.value
        this.computeSurface()
    }

    onOptionTypeChange(event: any) {
        this.parameters.optionType = event.detail.value
        this.computeSurface()
    }


    onRangeChange(event: any) {
        this.parameters.range = event.target.value
        this.computeSurface()
    }

    loadOptions() {
        console.log(this.parameters)
        this.rowData$ = this.optionsService.getOptions(this.parameters.ticker)
        this.optionsService.getSpot(this.parameters.ticker).subscribe({
            next: d => {
                this.spot = d
                this.filterGridColumns()
                this.computeSurface()
            }
        })
    }

    filterGridColumns() {
        console.log(`Filtering grid for type ${this.parameters.optionType}`)
        this.gridOptions?.api?.getFilterInstance('strike')?.setModel({
            type: 'inRange',
            filter: Math.round(this.spot * (1 - this.parameters.range / 100)),
            filterTo: Math.round(this.spot * (1 + this.parameters.range / 100)),
        });
        this.gridOptions?.api?.getFilterInstance('type')?.setModel({ filterType: 'text', type: 'startsWith', filter: this.parameters.optionType });
        this.gridOptions?.api?.onFilterChanged();
    }


    computeSurface() {
        this.optionsService.getSurface(this.parameters.ticker, this.parameters.optionType, this.parameters.field, this.parameters.range).subscribe({

            next: d => {
                const zAxis = this.titlecasePipe.transform(this.parameters.field.replace("_", " "))
                const nbDecimals = this.parameters.field === "price" ? 2 : 5
                var data_update = {
                    'x': [d?.x], 'y': [d.y], 'z': [d.z], hovertemplate: 'Strike: %{y:.2f}<br>Maturity: %{x}<br>' + zAxis + ': %{z:.' + nbDecimals + 'f}',
                    colorscale: 'Portland'
                } as Plotly.Data;
                var layout_update = { 'scene.zaxis.title': zAxis };
                Plotly.update('divSurfaceOptions', data_update, layout_update, 0);
                this.filterGridColumns()
            }
        })
    }
}
