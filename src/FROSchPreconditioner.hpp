#ifndef FVM_FROSCH_PRECONDITIONER_H
#define FVM_FROSCH_PRECONDITIONER_H

#include <Xpetra_EpetraCrsMatrix.hpp>
#include "Ifpack_Preconditioner.h"
#include "FROSch_TwoLevelBlockPreconditioner_def.hpp"

namespace FROSch
{

    class IfpackPreconditioner : public Ifpack_Preconditioner
    {

        public:

            IfpackPreconditioner(Teuchos::RCP<const Epetra_RowMatrix> matrix,
                                 Teuchos::RCP<Teuchos::ParameterList> &parameterList);

            virtual ~IfpackPreconditioner() {}

            int SetParameters(Teuchos::ParameterList &List) { return 0; }

            int Initialize() {FROSCH_ASSERT(false,"not implemented"); return 0;};

            int InitializeNew(Teuchos::RCP<Epetra_Map> repeated_velocity_map,
                           Teuchos::RCP<Epetra_Map> u_map,
                           Teuchos::RCP<Epetra_Map> v_map,
                           Teuchos::RCP<Epetra_Map> w_map,
                           Teuchos::RCP<Epetra_Map> p_map,
                           Teuchos::RCP<Epetra_Map> t_map);

            bool IsInitialized() const { return IsInitialized_; }

            int Compute();

            bool IsComputed() const { return IsComputed_; }

            double Condest(const Ifpack_CondestType CT = Ifpack_Cheap,
                           const int MaxIters = 1550,
                           const double Tol = 1e-9,
                           Epetra_RowMatrix* Matrix = 0) { return 0.0; }

            double Condest() const { return 0.0; }

            int Apply(const Epetra_MultiVector &X,
                      Epetra_MultiVector &Y) const { return -1; }

            int ApplyInverse(const Epetra_MultiVector &X,
                             Epetra_MultiVector &Y) const;

            const Epetra_RowMatrix &Matrix() const { return *Matrix_; }

            int NumInitialize() const { return 0; }

            int NumCompute() const { return 0; }

            int NumApplyInverse() const { return 0; }

            double InitializeTime() const { return 0; }

            double ComputeTime() const { return 0; }

            double ApplyInverseTime() const { return 0; }

            double InitializeFlops() const { return 0; }

            double ComputeFlops() const { return 0; }

            double ApplyInverseFlops() const { return 0; }

            std::ostream &Print(std::ostream &os) const { return os; }

            int SetUseTranspose(bool UseTranspose) { return -1; }

            bool HasNormInf() const { return false; }

            double NormInf() const { return 0; }

            const char *Label() const { return "FROSchPreconditioner"; }

            bool UseTranspose() const { return false; }

            const Epetra_Comm &Comm() const { return Matrix_->Comm(); }

            const Epetra_Map &OperatorDomainMap() const { return Matrix_->OperatorDomainMap(); }

            const Epetra_Map &OperatorRangeMap() const { return Matrix_->OperatorRangeMap(); }

        protected:
            Teuchos::RCP<const Epetra_Comm> Comm_;
            Teuchos::RCP<const Teuchos::Comm<int> > TeuchosComm_ = Teuchos::null;

            Teuchos::RCP<const Epetra_RowMatrix> Matrix_;
            Teuchos::RCP<Teuchos::ParameterList> ParameterList_;
            Teuchos::RCP<TwoLevelBlockPreconditioner<double,int> > FROSchPreconditioner_ = Teuchos::null;

            bool IsInitialized_ = false;
            bool IsComputed_ = false;
    };

}  // namespace FROSch

#include "FROSchPreconditioner.cpp"

#endif
