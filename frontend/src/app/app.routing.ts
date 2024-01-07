import { Routes } from '@angular/router';
import { ExplorerDashboardComponent } from './modules/pages/dashboard/explorer.dashboard.component';



export const AppRoutes: Routes = [
  { path: 'explorer', component: ExplorerDashboardComponent },
  { path: "**", redirectTo: "explorer" }
]
